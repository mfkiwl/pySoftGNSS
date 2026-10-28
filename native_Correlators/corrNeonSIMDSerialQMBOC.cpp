/*================================================================================
 * Filename: corrNeonSIMDSerialQMBOC.cpp
 * Description: AArch64 NEON SIMD single-channel QMBOC tracking correlator exposed
 *              through a C interface for Python ctypes.
 *
 * Authors: Yafeng Li (School of Automation, Beijing Information Science and Technology University)
 * Time: Aug, 1, 2026
 *
 * Processing stages:
 *   1) mixCarrReal() / mixCarrComplex()
 *   2) localCodeGen()
 *   3) correlator()
 *
 * This file processes one channel and one code period per call. The first
 * corrEngine() call allocates reusable work buffers; corrEngineFree() releases
 * them after channel-serial tracking has finished. Processing and cleanup
 * calls share state and must be serialized.
 *==============================================================================*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <arm_neon.h>

/* Disable implicit floating-point multiply-add contraction;
 * explicit NEON FMA intrinsics remain fused. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

/* Use C linkage and platform-specific attributes to export the entry points. */
#ifdef _WIN32
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* Settings struct -------------------------------------------------------------
 * This structure stores the receiver settings cached by the SIMD correlator.
 * fileType       : 1 - real samples; 2 - interleaved complex samples
 * rShiftBits     : number of right-shift bits for IF data to prevent SIMD
 *                  correlator overflow when ADC valid bits occupy the high bits
 * earlyLateSpc remains a per-call corrEngine() argument.
 *------------------------------------------------------------------------------*/
typedef struct
{
	int fileType;        /* 1 - real samples; 2 - complex samples */
	int rShiftBits;      /* Right-shift bits for IF data before SIMD processing */
} Settings;

/* Function declaration ---------------------------------------------------------
 * mixCarrReal()        : mix local carrier with real IF data
 * mixCarrComplex()     : mix local carrier with complex IF data
 * localCodeGen()       : generate data/pilot BOC(1,1)/pilot BOC(6,1) EPL local code sequences
 * correlator()         : compute one branch of IE/QE/IP/QP/IL/QL correlation outputs
 * cleanup()            : release persistent host buffers
 * checkedAlignedMalloc(): allocate 16-byte aligned host memory for SIMD buffers
 * gatherCode16()       : gather 16 local-code samples and repack them to int16 order
 * checkInputs()        : validate the fixed C ABI inputs on the first corrEngine() call
 *------------------------------------------------------------------------------*/
void mixCarrReal(const short*, double, double, size_t, short*, short*);
void mixCarrComplex(const short*, double, double, size_t, short*, short*);
void localCodeGen(const int*, size_t, double, double, double, size_t, short**);
void correlator(short*, short*, short**, size_t, double*);
void cleanup(void);
static int checkInputs(int fileType, int blksize, const short* rawSignal,
	const int* caCode, int codeLen);
static void* checkedAlignedMalloc(size_t nbytes, const char* name);
static int16x8x2_t gatherCode16(const int* caCode, int32x4x2_t Index_reg1, int32x4x2_t Index_reg2);

/* Constants -------------------------------------------------------------------- */
#define PI          3.1415926535897932  // pi 
#define DPI         (2.0*PI)            // 2*pi 
#define CSCALE      (1.0/16.0)          // carrier lookup table scale (LSB) 
#define LOCAL_CODE_COUNT 9              // 3 branches x Early/Prompt/Late

/* Persistent buffers -----------------------------------------------------------
 * initialized     : initialization flag for persistent buffers
 * iBasebandSignal : in-phase baseband samples after carrier wipeoff
 * qBasebandSignal : quadrature baseband samples after carrier wipeoff
 * localCode       : local code buffers for data/pilot BOC(1,1)/pilot BOC(6,1) EPL replicas
 * settings        : file type and input right shift cached on the first call
 *------------------------------------------------------------------------------*/
static int initialized = 0;
static short* iBasebandSignal = NULL;
static short* qBasebandSignal = NULL;
static short* localCode[LOCAL_CODE_COUNT] = { NULL };
static Settings settings;

/* Python ctypes entry point ---------------------------------------------------
 * Args   : fileType       I   1 - real samples; 2 - interleaved I/Q samples
 *          blksize        I   physical number of int16 elements in rawSignal
 *          rShiftBits     I   input right shift used by the SIMD mixer
 *          earlyLateSpc   I   half early-late spacing in code-table grid units
 *          rawSignal      I   int16 IF samples
 *          caCode         I   concatenated int32 guarded code branches
 *                              [B1CData, pilotBOC11, pilotBOC61]
 *          codeLen        I   guarded code length of one branch; caCode contains
 *                              3 * codeLen elements
 *          remCarrPhase   I   residual carrier phase (rad)
 *          carrPhaseStep  I   carrier phase increment (rad/sample)
 *          remCodePhase   I   residual code phase in code-table grid units
 *          codePhaseStep  I   code-table grid increment per IF sample
 * Return : pointer to static [data E/P/L I/Q, pilotBOC11 E/P/L I/Q,
 *          pilotBOC61 E/P/L I/Q], or NULL on first-call validation/allocation
 *          failure. The array contents are overwritten by the next call.
 * Notes  : Python supplies code phase, step and spacing on the common 12x
 *          primary-code grid; this function does not scale them again.
 *------------------------------------------------------------------------------*/
DLL_EXPORT double* corrEngine(int fileType, int blksize, int rShiftBits,
	double earlyLateSpc, short* rawSignal, int* caCode, int codeLen,
	double remCarrPhase, double carrPhaseStep, double remCodePhase,
	double codePhaseStep)
{
	static double corrValues[2 * LOCAL_CODE_COUNT] = { 0.0 };

	/* --------------------- One-time persistent buffer allocation ---------------------
	   Allocate once on the first call and reuse the same aligned buffers on later
	   code periods. Later logical block sizes must fit the first size + 100. */
	if (!initialized) {
		/* ----------------- Validate and cache the serial settings once -----------------
		   Cached settings and input types remain fixed until corrEngineFree(). */
		// gateway check
		if (!checkInputs(fileType, blksize, rawSignal, caCode, codeLen)) {
			return NULL;
		}
		// Basic setting
		settings.fileType = fileType;
		settings.rShiftBits = rShiftBits;

		size_t rawSignalLen = (size_t)blksize;
		if (settings.fileType == 2) {
			rawSignalLen = rawSignalLen / 2;
		}
		rawSignalLen = rawSignalLen + 100;
		iBasebandSignal = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "iBasebandSignal");
		qBasebandSignal = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "qBasebandSignal");

		for (int i = 0; i < LOCAL_CODE_COUNT; ++i) {
			localCode[i] = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "localCode[i]");
		}

		if (iBasebandSignal == NULL || qBasebandSignal == NULL ||
			localCode[0] == NULL || localCode[1] == NULL || localCode[2] == NULL ||
			localCode[3] == NULL || localCode[4] == NULL || localCode[5] == NULL ||
			localCode[6] == NULL || localCode[7] == NULL || localCode[8] == NULL) {
			cleanup();
			return NULL;
		}

		initialized = 1;
		printf("   SIMD DLL initialized ...\n");
		printf("   Channel-serial tracking with the SIMD-accelerated correlator developed by Yafeng Li.\n");
	}

	/* fileType and rShiftBits are fixed for the lifetime of the initialized
	   engine. earlyLateSpc and all phase parameters remain per-call inputs. */
	if (settings.fileType == 2) {
		/* The C ABI counts int16 elements; the three stages count complex samples. */
		blksize = blksize / 2;
	}

	/* --------------------- Initialize output and run the three-stage core ---------------------
	   Output layout:
	     [data E/P/L I/Q, pilotBOC11 E/P/L I/Q, pilotBOC61 E/P/L I/Q] */
	if (settings.fileType == 1) {
		mixCarrReal(rawSignal, carrPhaseStep, remCarrPhase, (size_t)blksize, iBasebandSignal, qBasebandSignal);
	} else {
		mixCarrComplex(rawSignal, carrPhaseStep, remCarrPhase, (size_t)blksize, iBasebandSignal, qBasebandSignal);
	}
	localCodeGen(caCode, (size_t)codeLen, remCodePhase, codePhaseStep, earlyLateSpc, (size_t)blksize, localCode);
	// For data branch
	correlator(iBasebandSignal, qBasebandSignal, localCode, (size_t)blksize, corrValues);
	// For pilot BOC(1,1) branch
	correlator(iBasebandSignal, qBasebandSignal, localCode+3, (size_t)blksize, corrValues+6);
	// For pilot BOC(6,1) branch
	correlator(iBasebandSignal, qBasebandSignal, localCode+6, (size_t)blksize, corrValues+12);
	return corrValues;
}

/* corrEngineFree -------------------------------------------------------------
 * Explicitly release persistent resources from the Python close/finally path.
 * It is safe to call repeatedly.
 *------------------------------------------------------------------------------*/
DLL_EXPORT void corrEngineFree(void)
{
	cleanup();
}

/* Mix local carrier for real data -----------------------------------------------
 * Mix local carrier to input signal
 * Args   : short   *rawSignal         I   Input IF signal
 *          double carrPhaseStep       I   carrier phase increment (rad/sample)
 *          double remCarrPhase        I   initial carrier phase (rad)
 *          size_t blksize             I   number of real IF samples
 *          short  *iBasebandSignal    O   I component of input signal with carrier wiped off 
 *          short  *qBasebandSignal    O   Q component of input signal with carrier wiped off
 * Return : None
 *------------------------------------------------------------------------------*/
void mixCarrReal(const short *rawSignal, double carrPhaseStep, double remCarrPhase, size_t blksize,
	         short *iBasebandSignal, short *qBasebandSignal)
{
	/* ---------- Inphase and quadrature carrier lookup table --------- */
	static int8_t cost[16] = { 0 }, sint[16] = { 0 };
	//Carrier lookup table for exp(-j*carrPhase)
	/* Quantize 16*cos(phase) and -16*sin(phase) using floor(x + 0.5). */
	if (!cost[0]) {
        for (int i = 0; i < 16; ++i) {
            cost[i] = (int8_t)floor(cos(PI / 8.0 * i) / CSCALE + 0.5);
            sint[i] = (int8_t)floor(-sin(PI / 8.0 * i) / CSCALE + 0.5);
		}
	}

	/* Convert radians to a 16-state phase accumulator used by the LUT.
	   One integer step corresponds to pi/8, so phase indexes are simply
	   obtained by rounding the scaled phase and masking with 0x0F. */
    remCarrPhase = remCarrPhase * 16.0 / DPI;
	/* Carrier phase step */
    const double phaseStep = carrPhaseStep * 16.0 / DPI;

	/* Paired 128-bit vectors hold each block of 16 int16 samples. */
	int16x8x2_t Index16_reg, cos_reg, sin_reg;
	int32x4x2_t Index_reg1, Index_reg2;
	uint8x16_t Index_reg;
	int8x16_t cos8_reg, sin8_reg;

	/* The chosen step-number order makes the low/high eight-sample groups
	   correspond to samples 0..7 and 8..15 after narrowing and combining. */
	const float stepNumber[8] = { 0.0f, 1.0f, 2.0f, 3.0f, 8.0f, 9.0f, 10.0f, 11.0f };
	float32x4x2_t stepNumber_reg = { { vld1q_f32(stepNumber), vld1q_f32(stepNumber + 4) } };
	const float32x4_t remCarrPhase_reg = vdupq_n_f32((float)remCarrPhase);
	const float32x4_t carrPhaseStep_reg = vdupq_n_f32((float)phaseStep);
	const int32x4_t mask4 = vdupq_n_s32(15);
	const float32x4_t sixteenStep = vdupq_n_f32(16.0f);
	const float32x4_t fourPhaseStep = vdupq_n_f32((float)(phaseStep * 4.0));
	const int8x16_t local_cos = vld1q_s8(cost);
	const int8x16_t local_sin = vld1q_s8(sint);

	const size_t simdSamples = blksize / 16 * 16;
	size_t sampleIndex = 0;
    const short *IFData = rawSignal;
	for (; sampleIndex < simdSamples; sampleIndex += 16, IFData += 16, iBasebandSignal += 16, qBasebandSignal += 16) 
	{
		/* ------------------ Inphase and quadrature carriers generation ------------------ */
		/* Generate the carrier phase of 16 consecutive samples. Two four-lane
		   phase vectors cover eight samples per k iteration; both iterations
		   cover the 16-sample block. An explicit FMA is followed by a separate
		   four-step addition. */
		for (int k = 0; k < 2; ++k) {
			const float32x4_t carrPhase_reg1 = vfmaq_f32(remCarrPhase_reg, carrPhaseStep_reg, stepNumber_reg.val[k]);
			const float32x4_t carrPhase_reg2 = vaddq_f32(carrPhase_reg1, fourPhaseStep);

			/* Convert phase to LUT indexes. vcvtnq rounds float32 phases to nearest-even;
			   truncation can increase quantization error. Mask indexes with 0x0F. */
			Index_reg1.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg1), mask4);
			Index_reg2.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg2), mask4);
        }

		/* Narrow 16 int32 indexes to int16, then to 16 contiguous bytes.
		   The indexes are already 0..15, so ordinary narrowing preserves
		   their values and the groups combine into time order 0..15. */
		Index16_reg.val[0] = vcombine_s16(vmovn_s32(Index_reg1.val[0]), vmovn_s32(Index_reg2.val[0]));
		Index16_reg.val[1] = vcombine_s16(vmovn_s32(Index_reg1.val[1]), vmovn_s32(Index_reg2.val[1]));
		/* Lookup indexes use unsigned bytes; reinterpret keeps all bits unchanged. */
		Index_reg = vreinterpretq_u8_s8(vcombine_s8(
			vmovn_s16(Index16_reg.val[0]), vmovn_s16(Index16_reg.val[1])));

		/* Look up 16 quantized carrier samples from the 16-entry cosine/sine LUTs.
		   Each byte in Index_reg selects one byte from local_cos/local_sin. */
		cos8_reg = vqtbl1q_s8(local_cos, Index_reg);
		sin8_reg = vqtbl1q_s8(local_sin, Index_reg);
		/* Expand carrier samples to int16 so they can be multiplied directly with
		   the int16 IF samples. Carrier amplitude is quantized by CSCALE = 1/16. */
		cos_reg.val[0] = vmovl_s8(vget_low_s8(cos8_reg));
		cos_reg.val[1] = vmovl_s8(vget_high_s8(cos8_reg));
		sin_reg.val[0] = vmovl_s8(vget_low_s8(sin8_reg));
		sin_reg.val[1] = vmovl_s8(vget_high_s8(sin8_reg));

		/* ----------------- Carrier wiping off from the input IF signals ----------------- */
		for (int k = 0; k < 2; ++k) {
			/* Load eight real IF samples for this half of the 16-sample block. */
			int16x8_t IF_reg = vld1q_s16(IFData + 8 * k);
			/* Apply the caller's arithmetic right shift before carrier mixing.
			   rShiftBits provides ADC alignment and int16/int32 headroom. */
			if (settings.rShiftBits != 0) {
				IF_reg = vshlq_s16(IF_reg, vdupq_n_s16((short)-settings.rShiftBits));
			}

			/* For selected LUT coefficients C and S, form Ibb = x*C and Qbb = x*S.
			   C approximates 16*cos(phase); S approximates -16*sin(phase).
			   Int16 multiplication retains the low 16 bits without saturation. */
			const int16x8_t iBaseband_reg = vmulq_s16(IF_reg, cos_reg.val[k]);
			const int16x8_t qBaseband_reg = vmulq_s16(IF_reg, sin_reg.val[k]);
			/* Store the de-rotated baseband samples for correlator(). */
			vst1q_s16(iBasebandSignal + 8 * k, iBaseband_reg);
			vst1q_s16(qBasebandSignal + 8 * k, qBaseband_reg);

			/* Advance the synthetic sample index by 16 samples for the next SIMD block. */
			stepNumber_reg.val[k] = vaddq_f32(stepNumber_reg.val[k], sixteenStep);
		}
	}	

	/* ----------------- For sampling points outside the SIMD iteration  ----------------- */
	/* Reconstruct the double carrier phase at the first leftover sample.
	   Advance the phase in double precision and truncate LUT indexes toward
	   zero, rather than using SIMD float32 nearest-even rounding. */
	remCarrPhase = simdSamples * phaseStep + remCarrPhase;
	for (; sampleIndex < blksize; ++sampleIndex, ++IFData, ++iBasebandSignal, ++qBasebandSignal) {
        short sample = IFData[0];
        if (settings.rShiftBits != 0) {
            sample = (short)(sample >> settings.rShiftBits);
        }
		/* Wrap the truncated phase to 0..15, including negative phases. */
		int n = ((int)remCarrPhase) & 0x0F;
        iBasebandSignal[0] = (short)(cost[n] * sample);
        qBasebandSignal[0] = (short)(sint[n] * sample);
        remCarrPhase += phaseStep;
	}
}

/* Mix local carrier for complex data --------------------------------------------
 * Mix local carrier to input signal
 * Args   : short   *rawSignal         I   Interleaved complex IF samples [I0, Q0, I1, Q1, ...]
 *          double carrPhaseStep       I   carrier phase increment (rad/sample)
 *          double remCarrPhase        I   Initial carrier phase (rad)
 *          size_t blksize             I   number of complex IF samples
 *          short  *iBasebandSignal    O   I component of input signal with carrier wiped off
 *          short  *qBasebandSignal    O   Q component of input signal with carrier wiped off
 * Return : None
 *------------------------------------------------------------------------------*/
void mixCarrComplex(const short* rawSignal, double carrPhaseStep, double remCarrPhase, size_t blksize,
	short* iBasebandSignal, short* qBasebandSignal)
{
	/* Inphase and quadrature carrier lookup table */
	static int8_t cost[16] = { 0 }, sint[16] = { 0 };

	//Carrier lookup table for exp(-j*carrPhase)
	/* Quantize 16*cos(phase) and 16*sin(phase) using floor(x + 0.5). */
	if (!cost[0]) 	{
		for (int i = 0; i < 16; ++i) {
			cost[i] = (int8_t)floor(cos(PI / 8.0 * i) / CSCALE + 0.5);
			sint[i] = (int8_t)floor(sin(PI / 8.0 * i) / CSCALE + 0.5);
		}
	}

	/* Complex carrier wipeoff uses the same 16-state LUT idea as the real-data
	   version. Here sint[] stores +sin coefficients, and the signs are introduced later by
	   the complex multiply formulas for exp(-j*phase). */
    remCarrPhase = remCarrPhase * 16.0 / DPI;
	/* Carrier phase step */
    const double phaseStep = carrPhaseStep * 16.0 / DPI;

	int16x8x2_t Index16_reg, cos_reg, sin_reg;
	int32x4x2_t Index_reg1, Index_reg2;
	uint8x16_t Index_reg;
	int8x16_t cos8_reg, sin8_reg;

	/* Same index order as mixCarrReal(): the phase groups [0..3 | 8..11]
	   and [4..7 | 12..15] compact to samples 0, 1, 2, ..., 15. */
	const float stepNumber[8] = { 0.0f, 1.0f, 2.0f, 3.0f, 8.0f, 9.0f, 10.0f, 11.0f };
	float32x4x2_t stepNumber_reg = { { vld1q_f32(stepNumber), vld1q_f32(stepNumber + 4) } };
	const float32x4_t remCarrPhase_reg = vdupq_n_f32((float)remCarrPhase);
	const float32x4_t carrPhaseStep_reg = vdupq_n_f32((float)phaseStep);
	const int32x4_t mask4 = vdupq_n_s32(15);
	const float32x4_t sixteenStep = vdupq_n_f32(16.0f);
	const float32x4_t fourPhaseStep = vdupq_n_f32((float)(phaseStep * 4.0));
	const int8x16_t local_cos = vld1q_s8(cost);
	const int8x16_t local_sin = vld1q_s8(sint);

	const size_t simdSamples = blksize / 16 * 16;
	size_t sampleIndex = 0;
    const short *IFData = rawSignal;
    for (; sampleIndex < simdSamples; sampleIndex += 16, IFData += 32, iBasebandSignal += 16, qBasebandSignal += 16)
	{
		/* ------------------ Inphase and quadrature carriers generation ------------------ */
		/* Build the carrier phase for 16 adjacent complex samples. */
		/* Two four-lane phase vectors cover eight samples per k iteration.
		   Use an explicit FMA followed by a separate four-step addition. */
		for (int k = 0; k < 2; ++k) {
			const float32x4_t carrPhase_reg1 = vfmaq_f32(remCarrPhase_reg, carrPhaseStep_reg, stepNumber_reg.val[k]);
			const float32x4_t carrPhase_reg2 = vaddq_f32(carrPhase_reg1, fourPhaseStep);

			/* Convert those phases into 0..15 LUT indexes. */
			/* vcvtnq rounds to nearest-even; masking wraps indexes to 0..15. */
			Index_reg1.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg1), mask4);
			Index_reg2.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg2), mask4);
		}

		/* Narrow 16 int32 indexes to int16, then to 16 contiguous bytes.
		   The indexes are already 0..15, so ordinary narrowing preserves
		   their values and the groups combine into time order 0..15. */
		Index16_reg.val[0] = vcombine_s16(vmovn_s32(Index_reg1.val[0]), vmovn_s32(Index_reg2.val[0]));
		Index16_reg.val[1] = vcombine_s16(vmovn_s32(Index_reg1.val[1]), vmovn_s32(Index_reg2.val[1]));
		/* Lookup indexes use unsigned bytes; reinterpret keeps all bits unchanged. */
		Index_reg = vreinterpretq_u8_s8(vcombine_s8(
			vmovn_s16(Index16_reg.val[0]), vmovn_s16(Index16_reg.val[1])));

		/* Fetch the 16 cosine and sine coefficients from the quantized LUTs. */
		cos8_reg = vqtbl1q_s8(local_cos, Index_reg);
		sin8_reg = vqtbl1q_s8(local_sin, Index_reg);
		/* Expand LUT output to int16 for the complex multiply below. */
		cos_reg.val[0] = vmovl_s8(vget_low_s8(cos8_reg));
		cos_reg.val[1] = vmovl_s8(vget_high_s8(cos8_reg));
		sin_reg.val[0] = vmovl_s8(vget_low_s8(sin8_reg));
		sin_reg.val[1] = vmovl_s8(vget_high_s8(sin8_reg));

		/* ----------------- Carrier wiping off from the input IF signals ----------------- */
		/* Load each block as two groups of eight interleaved complex samples.
		   vld2q_s16 separates I/Q directly into vectors without split buffers. */
		for (int k = 0; k < 2; ++k) {
			const int16x8x2_t IF_reg = vld2q_s16(IFData + 16 * k);
			int16x8_t IF_regI = IF_reg.val[0];
			int16x8_t IF_regQ = IF_reg.val[1];
			/* Apply the caller's arithmetic right shift to both I/Q branches. */
			if (settings.rShiftBits != 0) {
				IF_regI = vshlq_s16(IF_regI, vdupq_n_s16((short)-settings.rShiftBits));
				IF_regQ = vshlq_s16(IF_regQ, vdupq_n_s16((short)-settings.rShiftBits));
			}

			/* Compute the four products needed for:
			     (I + jQ) * (cos - j sin)
			   and then combine them into the real/imag parts of the baseband signal.
			   Here cos/sin denote the quantized integer LUT coefficients.
			   Int16 operations retain the low 16 bits without saturation. */
			const int16x8_t sin_regI = vmulq_s16(IF_regI, sin_reg.val[k]);
			const int16x8_t cos_regI = vmulq_s16(IF_regI, cos_reg.val[k]);
			const int16x8_t sin_regQ = vmulq_s16(IF_regQ, sin_reg.val[k]);
			const int16x8_t cos_regQ = vmulq_s16(IF_regQ, cos_reg.val[k]);

			/* Combine the four components to real and imag part of the complex baseband */
			/* Ibb = I*cos + Q*sin; Qbb = Q*cos - I*sin,
			   using the quantized integer LUT coefficients. */
			const int16x8_t iBaseband_reg = vaddq_s16(cos_regI, sin_regQ);
			const int16x8_t qBaseband_reg = vsubq_s16(cos_regQ, sin_regI);
			/* Save the de-rotated complex baseband samples for the code correlator. */
			vst1q_s16(iBasebandSignal + 8 * k, iBaseband_reg);
			vst1q_s16(qBasebandSignal + 8 * k, qBaseband_reg);

			stepNumber_reg.val[k] = vaddq_f32(stepNumber_reg.val[k], sixteenStep);
		}
	}

	/* ----------------- For sampling points outside the SIMD iteration  ----------------- */
	/* Reconstruct the scalar phase state that follows the vectorized section.
	   Advance the phase in double precision and truncate LUT indexes toward
	   zero, rather than using SIMD float32 nearest-even rounding. */
	remCarrPhase = simdSamples * phaseStep + remCarrPhase;
	for (; sampleIndex < blksize; ++sampleIndex, IFData += 2, ++iBasebandSignal, ++qBasebandSignal)
	{
        short sampleI = IFData[0];
        short sampleQ = IFData[1];
        if (settings.rShiftBits != 0) {
            sampleI = (short)(sampleI >> settings.rShiftBits);
            sampleQ = (short)(sampleQ >> settings.rShiftBits);
        }
		/* Scalar fallback uses the same complex multiply as the SIMD path:
		     Ibb =  I*cos + Q*sin
		     Qbb =  Q*cos - I*sin
		   Here cos/sin denote the quantized integer LUT coefficients. */
		/* Each complete I/Q expression is converted to short. */
		/* Wrap the truncated phase to 0..15, including negative phases. */
		int n = ((int)remCarrPhase) & 0x0F;
        iBasebandSignal[0] = (short)(cost[n] * sampleI + sint[n] * sampleQ);
        qBasebandSignal[0] = (short)(cost[n] * sampleQ - sint[n] * sampleI);
		remCarrPhase += phaseStep;
	}
}

 /* Generate local code ----------------------------------------------------------
 * Generate local code sequences for the QMBOC data, pilot BOC(1,1), and
 * pilot BOC(6,1) branches.
 * Args   : int     *caCode          I   int32 concatenated QMBOC code table
 *                                      [guarded B1CData,
 *                                       guarded pilotBOC11,
 *                                       guarded pilotBOC61]
 *          size_t codeLen           I   element count of one guarded branch
 *          double remCodePhase      I   initial code phase in code-table grid units
 *          double codePhaseStep     I   code-table grid increment per IF sample
 *          double earlyLateSpc      I   half of the early-late code correlation spacing in code-table grid units
 *          size_t blksize           I   number of samples to be generated 
 *          short **localCode        O   local code replica outputs arranged as
 *                                      {data E/P/L, pilotBOC11 E/P/L, pilotBOC61 E/P/L}
 * return : None 
 *------------------------------------------------------------------------------*/
void localCodeGen(const int *caCode, size_t codeLen, double remCodePhase, double codePhaseStep,
	              double earlyLateSpc, size_t blksize, short **localCode)
{
	/* Initial code phase of early code: 1.0 is due to the leading guard
	value added to caCode for early replica generation */
	remCodePhase = remCodePhase - earlyLateSpc + 1.0;
	if (remCodePhase >= codeLen)
		remCodePhase -= floor(remCodePhase / codeLen)*codeLen;
	/* Keep the large integer part of the code phase out of SIMD float
	   arithmetic. Converting a large code-phase value to float can lose fractional
	   code-table grid precision. */
	const int codeIndexBase = (int)floor(remCodePhase);
	remCodePhase -= (double)codeIndexBase;

	/* caCode concatenates guarded data, pilotBOC11 and pilotBOC61 tables.
	   The caller supplies the guard-phase offset and enough guard points for
	   all E/P/L indexes; no per-sample wrapping is performed here.
	   Branches share E/P/L indexes; only the table base offset changes. */
	short *eCodeD = localCode[0], *pCodeD = localCode[1], * lCodeD = localCode[2];
	short* eCodeP11 = localCode[3], * pCodeP11 = localCode[4], * lCodeP11 = localCode[5];
	short* eCodeP61 = localCode[6], * pCodeP61 = localCode[7], * lCodeP61 = localCode[8];

	int32x4x2_t Index_reg1, Index_reg2;
	int16x8x2_t EPL_reg;
	float32x4x2_t codePhase_reg1, codePhase_reg2;
	/* Gather two groups of eight chronological indexes, then pack to int16. */
	const float stepNumber[8] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
	float32x4x2_t stepNumber_reg = { { vld1q_f32(stepNumber), vld1q_f32(stepNumber + 4) } };
	const float32x4_t remCodePhase_reg = vdupq_n_f32((float)remCodePhase);
	const float32x4_t codePhaseStep_reg = vdupq_n_f32((float)codePhaseStep);
	const float32x4_t earlyLateSpc_reg = vdupq_n_f32((float)earlyLateSpc);
	const float32x4_t sixteenStep = vdupq_n_f32(16.0f);
	const float32x4_t eightCodePhase = vdupq_n_f32((float)(codePhaseStep * 8.0));
	const int32x4_t codeIndexBase_reg = vdupq_n_s32(codeIndexBase);

	const size_t simdSamples = blksize / 16 * 16;
	size_t sampleIndex = 0;
	for (; sampleIndex < simdSamples; sampleIndex += 16,
        eCodeD += 16, pCodeD += 16, lCodeD += 16,
        eCodeP11 += 16, pCodeP11 += 16, lCodeP11 += 16,
        eCodeP61 += 16, pCodeP61 += 16, lCodeP61 += 16)
	{
		/* ----------------------- Early code generation ----------------------- */
		/* Generate the code phase of 16 successive samples. Convert the float32
		   phase to int32 by truncation, then add back the integer code-index base. */
		for (int k = 0; k < 2; ++k) {
			codePhase_reg1.val[k] = vfmaq_f32(remCodePhase_reg, codePhaseStep_reg, stepNumber_reg.val[k]);
			codePhase_reg2.val[k] = vaddq_f32(codePhase_reg1.val[k], eightCodePhase);
			Index_reg1.val[k] = vcvtq_s32_f32(codePhase_reg1.val[k]);
			Index_reg2.val[k] = vcvtq_s32_f32(codePhase_reg2.val[k]);
			Index_reg1.val[k] = vaddq_s32(Index_reg1.val[k], codeIndexBase_reg);
			Index_reg2.val[k] = vaddq_s32(Index_reg2.val[k], codeIndexBase_reg);
		}
		/* Gather 16 local-code chips directly from the int32 code table,
		   then pack them back to int16 with signed saturation for correlator(). */
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		vst1q_s16(eCodeD, EPL_reg.val[0]);
		vst1q_s16(eCodeD + 8, EPL_reg.val[1]);

		/* Pilot BOC(1,1) and BOC(6,1) early branches use the second and third code-table spans. */
		EPL_reg = gatherCode16(caCode + codeLen, Index_reg1, Index_reg2);
		vst1q_s16(eCodeP11, EPL_reg.val[0]);
		vst1q_s16(eCodeP11 + 8, EPL_reg.val[1]);
		EPL_reg = gatherCode16(caCode + 2 * codeLen, Index_reg1, Index_reg2);
		vst1q_s16(eCodeP61, EPL_reg.val[0]);
		vst1q_s16(eCodeP61 + 8, EPL_reg.val[1]);

		/* ----------------------- Prompt code generation ----------------------- */
		/* Prompt is Early shifted by +earlyLateSpc code-table points. */
		for (int k = 0; k < 2; ++k) {
			codePhase_reg1.val[k] = vaddq_f32(codePhase_reg1.val[k], earlyLateSpc_reg);
			codePhase_reg2.val[k] = vaddq_f32(codePhase_reg2.val[k], earlyLateSpc_reg);
			Index_reg1.val[k] = vcvtq_s32_f32(codePhase_reg1.val[k]);
			Index_reg2.val[k] = vcvtq_s32_f32(codePhase_reg2.val[k]);
			Index_reg1.val[k] = vaddq_s32(Index_reg1.val[k], codeIndexBase_reg);
			Index_reg2.val[k] = vaddq_s32(Index_reg2.val[k], codeIndexBase_reg);
		}
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		vst1q_s16(pCodeD, EPL_reg.val[0]);
		vst1q_s16(pCodeD + 8, EPL_reg.val[1]);

		EPL_reg = gatherCode16(caCode + codeLen, Index_reg1, Index_reg2);
		vst1q_s16(pCodeP11, EPL_reg.val[0]);
		vst1q_s16(pCodeP11 + 8, EPL_reg.val[1]);
		EPL_reg = gatherCode16(caCode + 2 * codeLen, Index_reg1, Index_reg2);
		vst1q_s16(pCodeP61, EPL_reg.val[0]);
		vst1q_s16(pCodeP61 + 8, EPL_reg.val[1]);
		/* ----------------------- Late code generation ----------------------- */
		/* Late is Prompt shifted once more by +earlyLateSpc, i.e. Early + 2*spacing.
		   Keep the two additions separate to preserve float32 rounding. */
		for (int k = 0; k < 2; ++k) {
			codePhase_reg1.val[k] = vaddq_f32(codePhase_reg1.val[k], earlyLateSpc_reg);
			codePhase_reg2.val[k] = vaddq_f32(codePhase_reg2.val[k], earlyLateSpc_reg);
			Index_reg1.val[k] = vcvtq_s32_f32(codePhase_reg1.val[k]);
			Index_reg2.val[k] = vcvtq_s32_f32(codePhase_reg2.val[k]);
			Index_reg1.val[k] = vaddq_s32(Index_reg1.val[k], codeIndexBase_reg);
			Index_reg2.val[k] = vaddq_s32(Index_reg2.val[k], codeIndexBase_reg);
		}
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		vst1q_s16(lCodeD, EPL_reg.val[0]);
		vst1q_s16(lCodeD + 8, EPL_reg.val[1]);

		EPL_reg = gatherCode16(caCode + codeLen, Index_reg1, Index_reg2);
		vst1q_s16(lCodeP11, EPL_reg.val[0]);
		vst1q_s16(lCodeP11 + 8, EPL_reg.val[1]);
		EPL_reg = gatherCode16(caCode + 2 * codeLen, Index_reg1, Index_reg2);
		vst1q_s16(lCodeP61, EPL_reg.val[0]);
		vst1q_s16(lCodeP61 + 8, EPL_reg.val[1]);

		/* Move to the next batch of 16 code samples. */
		for (int k = 0; k < 2; ++k)
			stepNumber_reg.val[k] = vaddq_f32(stepNumber_reg.val[k], sixteenStep);
	}

	/* ------------ For sampling points outside the SIMD iteration  ------------ */
	/* Recreate the code phase at the first leftover sample after the SIMD loop.
	   Advance in double precision, truncate, then add the integer code-index base.
	   Scalar chips use short conversion rather than SIMD signed saturation. */
	double tempCodePhase = simdSamples * codePhaseStep + remCodePhase;
	double twoEarlyLateSpc = earlyLateSpc * 2;
	for (; sampleIndex < blksize; ++sampleIndex,
        ++eCodeD, ++pCodeD, ++lCodeD,
        ++eCodeP11, ++pCodeP11, ++lCodeP11,
        ++eCodeP61, ++pCodeP61, ++lCodeP61)
	{
		/* Scalar tail keeps the same data/pilot Early/Prompt/Late geometry as the SIMD path. */
		eCodeD[0] = caCode[codeIndexBase + (int)tempCodePhase];
		pCodeD[0] = caCode[codeIndexBase + (int)(tempCodePhase + earlyLateSpc)];
		lCodeD[0] = caCode[codeIndexBase + (int)(tempCodePhase + twoEarlyLateSpc)];
		eCodeP11[0] = caCode[codeLen + codeIndexBase + (int)tempCodePhase];
		pCodeP11[0] = caCode[codeLen + codeIndexBase + (int)(tempCodePhase + earlyLateSpc)];
		lCodeP11[0] = caCode[codeLen + codeIndexBase + (int)(tempCodePhase + twoEarlyLateSpc)];
		eCodeP61[0] = caCode[2 * codeLen + codeIndexBase + (int)tempCodePhase];
		pCodeP61[0] = caCode[2 * codeLen + codeIndexBase + (int)(tempCodePhase + earlyLateSpc)];
		lCodeP61[0] = caCode[2 * codeLen + codeIndexBase + (int)(tempCodePhase + twoEarlyLateSpc)];
		tempCodePhase += codePhaseStep;
	}
}

/* Correlating function ------------------------------------------------------------
 * Correlate one branch of local code replicas with the carrier-wiped baseband signal.
 * Args :   short  *iBasebandSignal    I   I component of input signal with carrier wiped off
 *          short  *qBasebandSignal    I   Q component of input signal with carrier wiped off
 *          short  **localCode         I   one three-replica branch {E, P, L}
 *          size_t blksize             I   number of baseband samples to correlate
 *          double *corrValues         O   outputs of one branch arranged as [I_E Q_E I_P Q_P I_L Q_L]
 * Return : None
 *------------------------------------------------------------------------------*/
void correlator(short *iBasebandSignal, short *qBasebandSignal, short **localCode, size_t blksize, double *corrValues)
{
	const short* pI = iBasebandSignal;
	const short* pQ = qBasebandSignal;
	short* eCode = localCode[0];
	short* pCode = localCode[1];
	short* lCode = localCode[2];

	/* Each local code sample is stored as int16. Adjacent products are added:
	     [a0*b0 + a1*b1, a2*b2 + a3*b3, ...]
	   so two NEON vectors hold eight int32 partial sums from 16 samples. */
	int16x8x2_t I_reg, Q_reg;
	int32x4x2_t I_E_reg, I_P_reg, I_L_reg, Q_E_reg, Q_P_reg, Q_L_reg;
	for (int k = 0; k < 2; ++k) {
		I_E_reg.val[k] = vdupq_n_s32(0);
		I_P_reg.val[k] = vdupq_n_s32(0);
		I_L_reg.val[k] = vdupq_n_s32(0);
		Q_E_reg.val[k] = vdupq_n_s32(0);
		Q_P_reg.val[k] = vdupq_n_s32(0);
		Q_L_reg.val[k] = vdupq_n_s32(0);
	}

	const size_t simdSamples = blksize / 16 * 16;
	size_t sampleIndex = 0;
	for (; sampleIndex < simdSamples; sampleIndex += 16, eCode += 16, 
	pCode += 16, lCode += 16, pI += 16, pQ += 16) 
	{
		/* --------------- Load IF signal with carrier wiped off --------------- */
		/* Read two groups of eight int16 samples. Baseband buffers and the
		   three local-code buffers are allocated with 16-byte alignment. */
		for (int k = 0; k < 2; ++k) {
			I_reg.val[k] = vld1q_s16(pI + 8 * k);
			Q_reg.val[k] = vld1q_s16(pQ + 8 * k);
		}

		/* --------------- Compute the early correlation value for this branch --------------- */
		for (int k = 0; k < 2; ++k) {
			const int16x8_t EPL_reg = vld1q_s16(eCode + 8 * k);
			/* Multiply samples by code chips and horizontally add adjacent pairs,
			   leaving int32 partial sums to accumulate in the NEON vector. */
			int32x4_t mul_reg = vpaddq_s32(
				vmull_s16(vget_low_s16(I_reg.val[k]), vget_low_s16(EPL_reg)),
				vmull_s16(vget_high_s16(I_reg.val[k]), vget_high_s16(EPL_reg)));
			I_E_reg.val[k] = vaddq_s32(I_E_reg.val[k], mul_reg);
			mul_reg = vpaddq_s32(
				vmull_s16(vget_low_s16(Q_reg.val[k]), vget_low_s16(EPL_reg)),
				vmull_s16(vget_high_s16(Q_reg.val[k]), vget_high_s16(EPL_reg)));
			Q_E_reg.val[k] = vaddq_s32(Q_E_reg.val[k], mul_reg);
		}

		/* --------------- Compute the prompt correlation value for this branch --------------- */
		for (int k = 0; k < 2; ++k) {
			const int16x8_t EPL_reg = vld1q_s16(pCode + 8 * k);
			/* Multiply samples by code chips and horizontally add adjacent pairs,
			   leaving int32 partial sums to accumulate in the NEON vector. */
			int32x4_t mul_reg = vpaddq_s32(
				vmull_s16(vget_low_s16(I_reg.val[k]), vget_low_s16(EPL_reg)),
				vmull_s16(vget_high_s16(I_reg.val[k]), vget_high_s16(EPL_reg)));
			I_P_reg.val[k] = vaddq_s32(I_P_reg.val[k], mul_reg);
			mul_reg = vpaddq_s32(
				vmull_s16(vget_low_s16(Q_reg.val[k]), vget_low_s16(EPL_reg)),
				vmull_s16(vget_high_s16(Q_reg.val[k]), vget_high_s16(EPL_reg)));
			Q_P_reg.val[k] = vaddq_s32(Q_P_reg.val[k], mul_reg);
		}

		/* --------------- Compute the late correlation value for this branch --------------- */
		for (int k = 0; k < 2; ++k) {
			const int16x8_t EPL_reg = vld1q_s16(lCode + 8 * k);
			/* Multiply samples by code chips and horizontally add adjacent pairs,
			   leaving int32 partial sums to accumulate in the NEON vector. */
			int32x4_t mul_reg = vpaddq_s32(
				vmull_s16(vget_low_s16(I_reg.val[k]), vget_low_s16(EPL_reg)),
				vmull_s16(vget_high_s16(I_reg.val[k]), vget_high_s16(EPL_reg)));
			I_L_reg.val[k] = vaddq_s32(I_L_reg.val[k], mul_reg);
			mul_reg = vpaddq_s32(
				vmull_s16(vget_low_s16(Q_reg.val[k]), vget_low_s16(EPL_reg)),
				vmull_s16(vget_high_s16(Q_reg.val[k]), vget_high_s16(EPL_reg)));
			Q_L_reg.val[k] = vaddq_s32(Q_L_reg.val[k], mul_reg);
		}

	}

	/* --------------- Integration for SIMD register elements ---------------
     * Reduce each four-lane int32 accumulator to an int64 sum, then combine
     * the two sums per output. Pairwise additions and accumulation in the
     * SIMD loop are non-saturating int32 operations; their final reduction
     * widens to int64.
     *---------------------------------------------------------------------- */
	long long I_E = vaddlvq_s32(I_E_reg.val[0]) + vaddlvq_s32(I_E_reg.val[1]);
	long long Q_E = vaddlvq_s32(Q_E_reg.val[0]) + vaddlvq_s32(Q_E_reg.val[1]);
	long long I_P = vaddlvq_s32(I_P_reg.val[0]) + vaddlvq_s32(I_P_reg.val[1]);
	long long Q_P = vaddlvq_s32(Q_P_reg.val[0]) + vaddlvq_s32(Q_P_reg.val[1]);
	long long I_L = vaddlvq_s32(I_L_reg.val[0]) + vaddlvq_s32(I_L_reg.val[1]);
	long long Q_L = vaddlvq_s32(Q_L_reg.val[0]) + vaddlvq_s32(Q_L_reg.val[1]);

	/* ----------------- For sampling points outside the SIMD iteration  ----------------- */
	/* Scalar cleanup directly accumulates the remaining 0..15 samples. */
	for (; sampleIndex < blksize; ++sampleIndex, ++pI, ++pQ, ++eCode, ++pCode, ++lCode) {
		I_E += (long long)pI[0] * eCode[0];
		Q_E += (long long)pQ[0] * eCode[0];
		I_P += (long long)pI[0] * pCode[0];
		Q_P += (long long)pQ[0] * pCode[0];
		I_L += (long long)pI[0] * lCode[0];
		Q_L += (long long)pQ[0] * lCode[0];
	}
	/* Convert one EPL branch's six I/Q outputs to double precision and undo the 16x
	   carrier LUT amplitude scaling introduced during carrier wipeoff. The caller
	   invokes this function once each for data, pilot BOC(1,1), and pilot BOC(6,1). */
	/* This does not reverse the caller's input right shift. */
    corrValues[0] = (double)I_E * CSCALE;
    corrValues[1] = (double)Q_E * CSCALE;
    corrValues[2] = (double)I_P * CSCALE;
    corrValues[3] = (double)Q_P * CSCALE;
    corrValues[4] = (double)I_L * CSCALE;
    corrValues[5] = (double)Q_L * CSCALE;
}

/* checkedAlignedMalloc --------------------------------------------------------
 * Allocate 16-byte aligned host memory for SIMD buffers.
 * Args   : size_t nbytes             I   number of bytes to allocate
 *          const char* name          I   buffer name for error reporting
 * Return : void*                        allocated aligned pointer, or NULL on failure
 *------------------------------------------------------------------------------*/
static void* checkedAlignedMalloc(size_t nbytes, const char* name)
{
	void* p = NULL;
	if (posix_memalign(&p, 16, nbytes) != 0) {
		printf("   Aligned memory allocation failed for %s.\n", name);
	}
	return p;
}

/* gatherCode16 ----------------------------------------------------------------
 * Gather 16 local-code chips from the int32 code table and repack them
 * into two NEON vectors of 8 x int16 samples for correlation.
 * Signed saturation narrows the gathered chips; all indexes must be in bounds.
 * Args   : const int* caCode         I   int32 code table for one PRN
 *          int32x4x2_t Index_reg1     I   code indexes for samples 0..7
 *          int32x4x2_t Index_reg2     I   code indexes for samples 8..15
 * Return : int16x8x2_t                   gathered code chips in chronological order
 *------------------------------------------------------------------------------*/
static int16x8x2_t gatherCode16(const int* caCode, int32x4x2_t Index_reg1, int32x4x2_t Index_reg2)
{
	/* Broadcast the first indexed chip into each four-lane vector, then
	   replace lanes 1..3 with their indexed chips. Keep int32 values until
	   the signed saturating conversion to int16 below. */
	int32x4x2_t EPL_reg1, EPL_reg2;
	for (int k = 0; k < 2; ++k) {
		EPL_reg1.val[k] = vld1q_dup_s32(caCode + vgetq_lane_s32(Index_reg1.val[k], 0));
		EPL_reg1.val[k] = vld1q_lane_s32(caCode + vgetq_lane_s32(Index_reg1.val[k], 1), EPL_reg1.val[k], 1);
		EPL_reg1.val[k] = vld1q_lane_s32(caCode + vgetq_lane_s32(Index_reg1.val[k], 2), EPL_reg1.val[k], 2);
		EPL_reg1.val[k] = vld1q_lane_s32(caCode + vgetq_lane_s32(Index_reg1.val[k], 3), EPL_reg1.val[k], 3);
		EPL_reg2.val[k] = vld1q_dup_s32(caCode + vgetq_lane_s32(Index_reg2.val[k], 0));
		EPL_reg2.val[k] = vld1q_lane_s32(caCode + vgetq_lane_s32(Index_reg2.val[k], 1), EPL_reg2.val[k], 1);
		EPL_reg2.val[k] = vld1q_lane_s32(caCode + vgetq_lane_s32(Index_reg2.val[k], 2), EPL_reg2.val[k], 2);
		EPL_reg2.val[k] = vld1q_lane_s32(caCode + vgetq_lane_s32(Index_reg2.val[k], 3), EPL_reg2.val[k], 3);
	}
	int16x8x2_t EPL_reg;
	EPL_reg.val[0] = vcombine_s16(vqmovn_s32(EPL_reg1.val[0]), vqmovn_s32(EPL_reg1.val[1]));
	EPL_reg.val[1] = vcombine_s16(vqmovn_s32(EPL_reg2.val[0]), vqmovn_s32(EPL_reg2.val[1]));
	return EPL_reg;
}

/* checkInputs ----------------------------------------------------------------
 * Check non-null pointers, positive lengths and supported fileType once,
 * before allocating persistent buffers. Other caller constraints are not checked.
 * Return : 1 if these basic checks pass; 0 otherwise
 *------------------------------------------------------------------------------*/
static int checkInputs(int fileType, int blksize, const short* rawSignal,
	const int* caCode, int codeLen)
{
	if (rawSignal == NULL || caCode == NULL || blksize <= 0 || codeLen <= 0 ||
		(fileType != 1 && fileType != 2)) {
		printf("   Invalid input arguments for corrNeonSIMDSerialQMBOC.\n");
		return 0;
	}
	return 1;
}

/* cleanup --------------------------------------------------------------------
 * Release all persistent host buffers when Python calls corrEngineFree().
 * It is safe to call repeatedly.
 * Return : None
 *------------------------------------------------------------------------------*/
void cleanup(void)
{
	printf("   SIMD DLL is terminating, destroying allocated memory ...\n");

	/* --------------- Free memory --------------- */
	for (int i = 0; i < LOCAL_CODE_COUNT; ++i) {
		free(localCode[i]);
		localCode[i] = NULL;
	}

	free(iBasebandSignal); iBasebandSignal = NULL;
	free(qBasebandSignal); qBasebandSignal = NULL;
	initialized = 0;
}
