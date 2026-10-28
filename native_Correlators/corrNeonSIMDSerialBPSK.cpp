/*================================================================================
 * Filename: corrNeonSIMDSerialBPSK.cpp
 * Description: AArch64 NEON SIMD single-channel tracking correlator exposed through
 *              a C interface for Python ctypes.
 *
 * Authors: Yafeng Li (School of Automation, Beijing Information Science and Technology University)
 * Time: Aug, 1, 2026
 *
 * Processing stages:
 *   1) mixCarrReal() / mixCarrComplex()
 *   2) localCodeGen()
 *   3) correlator()
 *
 * Each call processes one caller-supplied block for one channel. The first
 * corrEngine() call allocates reusable work buffers; corrEngineFree() releases
 * them. Buffers, settings and the result array are shared static state, so
 * processing and cleanup calls must be serialized.
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

/* Settings cached from the first call. fileType and rShiftBits remain fixed
 * during one tracking run. */
typedef struct
{
	int fileType;
	int rShiftBits;
} Settings;

/* Function declaration ---------------------------------------------------------
 * mixCarrReal()        : mix local carrier with real IF data
 * mixCarrComplex()     : mix local carrier with complex IF data
 * localCodeGen()       : generate early/prompt/late local code sequences
 * correlator()         : compute IE/QE/IP/QP/IL/QL correlation outputs
 * cleanup()            : release persistent host buffers
 * checkedAlignedMalloc(): allocate 16-byte aligned host memory for SIMD buffers
 * gatherCode16()       : gather 16 code chips and saturate them to int16 in time order
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
#define CSCALE      (1.0/16.0)          // normalized amplitude per integer carrier-LUT unit 

/* Persistent buffers -----------------------------------------------------------
 * initialized     : initialization flag for persistent buffers
 * iBasebandSignal : in-phase baseband samples after carrier wipeoff
 * qBasebandSignal : quadrature baseband samples after carrier wipeoff
 * localCode       : fixed E/P/L pointer table for three reusable code buffers
 * settings        : file type and input right shift cached on the first call
 *------------------------------------------------------------------------------*/
static int initialized = 0;
static short* iBasebandSignal = NULL;
static short* qBasebandSignal = NULL;
static short* localCode[3] = { NULL, NULL, NULL };
static Settings settings;

/* Python ctypes entry point ---------------------------------------------------
 * Args   : fileType       I   1 - real samples; 2 - interleaved I/Q samples
 *          blksize        I   int16 element count; must be even for interleaved I/Q
 *          rShiftBits     I   caller-selected arithmetic right shift of IF samples
 *          earlyLateSpc   I   half early-late spacing (chips)
 *          rawSignal      I   int16 IF samples
 *          caCode         I   int32 guarded code [last, one period, first]
 *          codeLen        I   number of elements in caCode
 *          remCarrPhase   I   residual carrier phase (rad)
 *          carrPhaseStep  I   carrier phase increment (rad/sample)
 *          remCodePhase   I   residual code phase (chips)
 *          codePhaseStep  I   code phase increment (chips/sample)
 * Return : pointer to a shared static double array [IE QE IP QP IL QL], or
 *          NULL when first-call validation or allocation fails. The next
 *          successful corrEngine() call overwrites the array contents.
 *------------------------------------------------------------------------------*/
DLL_EXPORT double* corrEngine(int fileType, int blksize, int rShiftBits,
	double earlyLateSpc, short* rawSignal, int* caCode, int codeLen,
	double remCarrPhase, double carrPhaseStep, double remCodePhase,
	double codePhaseStep)
{
	static double corrValues[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

	/* Allocate the first logical block size + 100 samples for each buffer.
	   The caller must keep later logical blocks within this capacity;
	   these buffers are not resized until the engine is freed and reinitialized. */
	if (!initialized) {
		/* ----------------- Validate and cache the serial settings once -----------------
		   Cached settings and input types remain fixed until corrEngineFree(). */
		if (!checkInputs(fileType, blksize, rawSignal, caCode, codeLen)) {
			return NULL;
		}
		
		settings.fileType = fileType;
		settings.rShiftBits = rShiftBits;
		
		size_t rawSignalLen = (size_t)blksize;
		if (settings.fileType == 2) {
			rawSignalLen = rawSignalLen / 2;
		}
		rawSignalLen = rawSignalLen + 100;
		iBasebandSignal = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "iBasebandSignal");
		qBasebandSignal = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "qBasebandSignal");

		for (int i = 0; i < 3; ++i) {
			localCode[i] = (short*)checkedAlignedMalloc(sizeof(short) * rawSignalLen, "localCode[i]");
		}

		if (iBasebandSignal == NULL || qBasebandSignal == NULL ||
			localCode[0] == NULL || localCode[1] == NULL || localCode[2] == NULL) {
			cleanup();
			return NULL;
		}

		initialized = 1;
		printf("   SIMD DLL initialized ...\n");
		printf("   Channel-serial tracking with the SIMD-accelerated correlator developed by Yafeng Li.\n");
	}

	if (settings.fileType == 2) {
		/* The C ABI counts int16 elements; the three stages count complex samples. */
		blksize = blksize / 2;
	}

	/* -------------------------- Run the three-stage core --------------------------
	   Mix the carrier, generate E/P/L codes, then fill corrValues in
	   [IE QE IP QP IL QL] order. */
	if (settings.fileType == 1) {
		mixCarrReal(rawSignal, carrPhaseStep, remCarrPhase, (size_t)blksize, iBasebandSignal, qBasebandSignal);
	} else {
		mixCarrComplex(rawSignal, carrPhaseStep, remCarrPhase, (size_t)blksize, iBasebandSignal, qBasebandSignal);
	}
	localCodeGen(caCode, (size_t)codeLen, remCodePhase, codePhaseStep, earlyLateSpc, (size_t)blksize, localCode);
	correlator(iBasebandSignal, qBasebandSignal, localCode, (size_t)blksize, corrValues);
	return corrValues;
}

/* corrEngineFree -------------------------------------------------------------
 * Release reusable work buffers and reset initialization, e.g. in Python finally.
 * It is safe to call repeatedly.
 *------------------------------------------------------------------------------*/
DLL_EXPORT void corrEngineFree(void)
{
	cleanup();
}

/* Mix local carrier for real data -----------------------------------------------
 * Remove the carrier from IF samples using quantized lookup coefficients.
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
	/* Quantize 16*cos(phase) and -16*sin(phase) using floor(x + 0.5).
	   This coefficient quantization is separate from phase-index rounding. */
	if (!cost[0]) {
		for (int i = 0; i < 16; ++i) {
			cost[i] = (int8_t)floor(cos(PI / 8.0 * i) / CSCALE + 0.5);
			sint[i] = (int8_t)floor(-sin(PI / 8.0 * i) / CSCALE + 0.5);
		}
	}

	/* Express phase in LUT states, with one state per pi/8 radians.
	   The SIMD path rounds float32 phases to nearest-even and masks
	   each integer index to 0..15. */
	remCarrPhase = remCarrPhase * 16.0 / DPI;
	/* Carrier phase increment in LUT states per sample. */
	const double phaseStep = carrPhaseStep * 16.0 / DPI;

	/* Paired 128-bit vectors hold each block of 16 int16 samples. */
	int16x8x2_t Index16_reg, cos_reg, sin_reg;
	int32x4x2_t Index_reg1, Index_reg2;
	float32x4x2_t carrPhase_reg1, carrPhase_reg2;
	uint8x16_t Index_reg;
	int8x16_t cos8_reg, sin8_reg;

	/* The first phase group covers samples [0..3 | 8..11]. Adding four
	   phase steps produces [4..7 | 12..15]. Narrowing and combining the
	   groups below restores chronological sample order 0, 1, ..., 15. */
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
	for (; sampleIndex < simdSamples; sampleIndex += 16, IFData += 16,
		iBasebandSignal += 16, qBasebandSignal += 16)
	{

		/* ------------------ Inphase and quadrature carriers generation ------------------ */
		/* Generate the carrier phase of 16 consecutive samples. Each NEON
		   vector holds four float32 phases. Use an explicit FMA followed
		   by a separate four-step addition for the second phase group. */
		for (int k = 0; k < 2; ++k) {
			carrPhase_reg1.val[k] = vfmaq_f32(remCarrPhase_reg, carrPhaseStep_reg, stepNumber_reg.val[k]);
			carrPhase_reg2.val[k] = vaddq_f32(carrPhase_reg1.val[k], fourPhaseStep);

			/* Round to nearest-even rather than truncating the phase, then
			   wrap the carrier index to the 16-entry LUT with a 0x0F mask. */
			Index_reg1.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg1.val[k]), mask4);
			Index_reg2.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg2.val[k]), mask4);
		}

		/* Narrow 16 int32 indexes to int16, then to 16 contiguous bytes.
		   The indexes are already 0..15, so ordinary narrowing preserves
		   their values and the groups combine into time order 0..15. */
		Index16_reg.val[0] = vcombine_s16(vmovn_s32(Index_reg1.val[0]), vmovn_s32(Index_reg2.val[0]));
		Index16_reg.val[1] = vcombine_s16(vmovn_s32(Index_reg1.val[1]), vmovn_s32(Index_reg2.val[1]));
		/* Lookup indexes use unsigned bytes; reinterpret keeps all bits unchanged. */
		Index_reg = vreinterpretq_u8_s8(vcombine_s8(
			vmovn_s16(Index16_reg.val[0]), vmovn_s16(Index16_reg.val[1])));

		/* Look up 16 quantized carrier samples from the cosine/sine LUTs.
		   Each byte in Index_reg selects one coefficient from each table. */
		cos8_reg = vqtbl1q_s8(local_cos, Index_reg);
		sin8_reg = vqtbl1q_s8(local_sin, Index_reg);
		/* Sign-extend the quantized int8 coefficients to int16.
		   One integer coefficient unit represents CSCALE = 1/16 in amplitude. */
		cos_reg.val[0] = vmovl_s8(vget_low_s8(cos8_reg));
		cos_reg.val[1] = vmovl_s8(vget_high_s8(cos8_reg));
		sin_reg.val[0] = vmovl_s8(vget_low_s8(sin8_reg));
		sin_reg.val[1] = vmovl_s8(vget_high_s8(sin8_reg));

		/* ----------------- Carrier wiping off from the input IF signals ----------------- */
		for (int k = 0; k < 2; ++k) {
			/* Load eight real IF samples for this half of the 16-sample block. */
			int16x8_t IF_reg = vld1q_s16(IFData + 8 * k);
			/* Arithmetic right shift reduces the IF level before mixing with
			   the 16x LUT coefficients. The caller selects rShiftBits for ADC
			   alignment and headroom in int16 mixing and int32 accumulation. */
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

			/* Advance the float32 sample offsets by 16 for the next block. */
			stepNumber_reg.val[k] = vaddq_f32(stepNumber_reg.val[k], sixteenStep);
		}
	}

	/* ----------------- Scalar tail: remaining blksize % 16 samples ----------------- */
	/* Reconstruct the double phase at the first leftover sample and advance
	   it in double precision. Each LUT index uses an int cast (truncation
	   toward zero) followed by & 0x0F, rather than SIMD nearest-even rounding. */
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
 * Remove the carrier from IF samples using quantized lookup coefficients.
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

	/* Quantize 16*cos(phase) and 16*sin(phase) using floor(x + 0.5).
	   The negative rotation sign is applied by the complex multiply below. */
	if (!cost[0]) {
		for (int i = 0; i < 16; ++i) {
			cost[i] = (int8_t)floor(cos(PI / 8.0 * i) / CSCALE + 0.5);
			sint[i] = (int8_t)floor(sin(PI / 8.0 * i) / CSCALE + 0.5);
		}
	}

	/* Complex carrier wipeoff uses the same 16-state quantization as the
	   real-data version, with one state per pi/8 radians. sint[] stores
	   +sin coefficients; exp(-j*phase) signs enter in the multiply below. */
	remCarrPhase = remCarrPhase * 16.0 / DPI;
	/* Carrier phase increment in LUT states per sample. */
	const double phaseStep = carrPhaseStep * 16.0 / DPI;

	int16x8x2_t Index16_reg, cos_reg, sin_reg;
	int32x4x2_t Index_reg1, Index_reg2;
	float32x4x2_t carrPhase_reg1, carrPhase_reg2;
	uint8x16_t Index_reg;
	int8x16_t cos8_reg, sin8_reg;

	/* The first phase group covers samples [0..3 | 8..11]. Adding four
	   phase steps produces [4..7 | 12..15]. Narrowing and combining the
	   groups below restores chronological sample order 0, 1, ..., 15. */
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
	for (; sampleIndex < simdSamples; sampleIndex += 16, IFData += 32,
		iBasebandSignal += 16, qBasebandSignal += 16)
	{

		/* ------------------ Inphase and quadrature carriers generation ------------------ */
		/* Generate the carrier phase of 16 consecutive samples. Each NEON
		   vector holds four float32 phases. Use an explicit FMA followed
		   by a separate four-step addition for the second phase group. */
		for (int k = 0; k < 2; ++k) {
			carrPhase_reg1.val[k] = vfmaq_f32(remCarrPhase_reg, carrPhaseStep_reg, stepNumber_reg.val[k]);
			carrPhase_reg2.val[k] = vaddq_f32(carrPhase_reg1.val[k], fourPhaseStep);

			/* Round to nearest-even rather than truncating the phase, then
			   wrap the carrier index to the 16-entry LUT with a 0x0F mask. */
			Index_reg1.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg1.val[k]), mask4);
			Index_reg2.val[k] = vandq_s32(vcvtnq_s32_f32(carrPhase_reg2.val[k]), mask4);
		}

		/* Narrow 16 int32 indexes to int16, then to 16 contiguous bytes.
		   The indexes are already 0..15, so ordinary narrowing preserves
		   their values and the groups combine into time order 0..15. */
		Index16_reg.val[0] = vcombine_s16(vmovn_s32(Index_reg1.val[0]), vmovn_s32(Index_reg2.val[0]));
		Index16_reg.val[1] = vcombine_s16(vmovn_s32(Index_reg1.val[1]), vmovn_s32(Index_reg2.val[1]));
		/* Lookup indexes use unsigned bytes; reinterpret keeps all bits unchanged. */
		Index_reg = vreinterpretq_u8_s8(vcombine_s8(
			vmovn_s16(Index16_reg.val[0]), vmovn_s16(Index16_reg.val[1])));

		/* Look up 16 quantized carrier samples from the cosine/sine LUTs.
		   Each byte in Index_reg selects one coefficient from each table. */
		cos8_reg = vqtbl1q_s8(local_cos, Index_reg);
		sin8_reg = vqtbl1q_s8(local_sin, Index_reg);
		/* Sign-extend the quantized int8 coefficients to int16.
		   One integer coefficient unit represents CSCALE = 1/16 in amplitude. */
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

			/* Compute four products with selected LUT coefficients C and S,
			   approximating 16*cos(phase) and 16*sin(phase), respectively.
			   Int16 multiplication and addition/subtraction retain the low
			   16 bits without saturation. */
			const int16x8_t sin_regI = vmulq_s16(IF_regI, sin_reg.val[k]);
			const int16x8_t cos_regI = vmulq_s16(IF_regI, cos_reg.val[k]);
			const int16x8_t sin_regQ = vmulq_s16(IF_regQ, sin_reg.val[k]);
			const int16x8_t cos_regQ = vmulq_s16(IF_regQ, cos_reg.val[k]);

			/* Combine the four products: Ibb = I*C + Q*S, Qbb = Q*C - I*S.
			   This applies the quantized negative carrier rotation. */
			const int16x8_t iBaseband_reg = vaddq_s16(cos_regI, sin_regQ);
			const int16x8_t qBaseband_reg = vsubq_s16(cos_regQ, sin_regI);
			/* Store the de-rotated baseband samples for correlator(). */
			vst1q_s16(iBasebandSignal + 8 * k, iBaseband_reg);
			vst1q_s16(qBasebandSignal + 8 * k, qBaseband_reg);

			stepNumber_reg.val[k] = vaddq_f32(stepNumber_reg.val[k], sixteenStep);
		}
	}

	/* ----------------- Scalar tail: remaining blksize % 16 samples ----------------- */
	/* Reconstruct the double phase at the first leftover sample and advance
	   it in double precision. Each LUT index uses an int cast (truncation
	   toward zero) followed by & 0x0F, rather than SIMD nearest-even rounding. */
	remCarrPhase = simdSamples * phaseStep + remCarrPhase;
	for (; sampleIndex < blksize; ++sampleIndex, IFData += 2, ++iBasebandSignal, ++qBasebandSignal)
	{
		short sampleI = IFData[0];
		short sampleQ = IFData[1];
		if (settings.rShiftBits != 0) {
			sampleI = (short)(sampleI >> settings.rShiftBits);
			sampleQ = (short)(sampleQ >> settings.rShiftBits);
		}
		/* Form the same complex multiply using cost[n] and sint[n] below;
		   each complete I/Q expression is converted to short. */
		/* Wrap the truncated phase to 0..15, including negative phases. */
		int n = ((int)remCarrPhase) & 0x0F;
		iBasebandSignal[0] = (short)(cost[n] * sampleI + sint[n] * sampleQ);
		qBasebandSignal[0] = (short)(cost[n] * sampleQ - sint[n] * sampleI);
		remCarrPhase += phaseStep;
	}
}

 /* Generate local code ----------------------------------------------------------
 * Generate local code sequences
 * Args   : int     *caCode          I   guarded int32 PRN table [last, one period, first]
 *          size_t codeLen           I   element count of the guarded caCode table
 *          double remCodePhase      I   initial code phase (chip)
 *          double codePhaseStep     I   code phase increment (chips/sample)
 *          double earlyLateSpc      I   half of the early-late code correlation spacing (chips)
 *          size_t blksize           I   number of samples to be generated 
 *          short **localCode        O   caller-allocated Early, Prompt, Late buffers
 * return : None 
 *------------------------------------------------------------------------------*/
void localCodeGen(const int *caCode, size_t codeLen, double remCodePhase, double codePhaseStep,
	              double earlyLateSpc, size_t blksize, short **localCode)
{
	/* Initial phase of the early replica: shift the prompt phase by
	   -earlyLateSpc. The +1 offset accounts for the leading guard chip. */
	remCodePhase = remCodePhase - earlyLateSpc + 1.0;
	if (remCodePhase >= codeLen)
		remCodePhase -= floor(remCodePhase / codeLen)*codeLen;

	/* caCode contains one code period with leading and trailing guard points.
	   The caller supplies the guard-phase offset and enough guard points for
	   all E/P/L indexes; no per-sample wrapping is performed here. */
	short *eCode = localCode[0], *pCode = localCode[1], *lCode = localCode[2];

	int32x4x2_t Index_reg1, Index_reg2;
	int16x8x2_t EPL_reg;
	float32x4x2_t codePhase_reg1, codePhase_reg2;
	/* The first group uses natural sample order 0..7 across two NEON
	   vectors; adding eight code-phase steps gives the group for 8..15. */
	const float stepNumber[8] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
	float32x4x2_t stepNumber_reg = { { vld1q_f32(stepNumber), vld1q_f32(stepNumber + 4) } };
	const float32x4_t remCodePhase_reg = vdupq_n_f32((float)remCodePhase);
	const float32x4_t codePhaseStep_reg = vdupq_n_f32((float)codePhaseStep);
	const float32x4_t earlyLateSpc_reg = vdupq_n_f32((float)earlyLateSpc);
	const float32x4_t sixteenStep = vdupq_n_f32(16.0f);
	const float32x4_t eightCodePhase = vdupq_n_f32((float)(codePhaseStep * 8.0));

	const size_t simdSamples = blksize / 16 * 16;
	size_t sampleIndex = 0;
	for (; sampleIndex < simdSamples; sampleIndex += 16, eCode += 16, pCode += 16, lCode += 16)
	{
		/* ----------------------- Early code generation ----------------------- */
		/* Generate the code phase of 16 successive samples using float32
		   FMA and a separate eight-step addition. Convert to int32 indexes
		   by truncation toward zero. */
		for (int k = 0; k < 2; ++k) {
			codePhase_reg1.val[k] = vfmaq_f32(remCodePhase_reg, codePhaseStep_reg, stepNumber_reg.val[k]);
			codePhase_reg2.val[k] = vaddq_f32(codePhase_reg1.val[k], eightCodePhase);
			Index_reg1.val[k] = vcvtq_s32_f32(codePhase_reg1.val[k]);
			Index_reg2.val[k] = vcvtq_s32_f32(codePhase_reg2.val[k]);
		}
		/* Gather 16 chips from the int32 table and saturate them to int16
		   in time order for the following correlation stage. */
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		vst1q_s16(eCode, EPL_reg.val[0]);
		vst1q_s16(eCode + 8, EPL_reg.val[1]);

		/* ----------------------- Prompt code generation ----------------------- */
		/* Prompt is Early shifted by +earlyLateSpc chips. */
		for (int k = 0; k < 2; ++k) {
			codePhase_reg1.val[k] = vaddq_f32(codePhase_reg1.val[k], earlyLateSpc_reg);
			codePhase_reg2.val[k] = vaddq_f32(codePhase_reg2.val[k], earlyLateSpc_reg);
			Index_reg1.val[k] = vcvtq_s32_f32(codePhase_reg1.val[k]);
			Index_reg2.val[k] = vcvtq_s32_f32(codePhase_reg2.val[k]);
		}
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		vst1q_s16(pCode, EPL_reg.val[0]);
		vst1q_s16(pCode + 8, EPL_reg.val[1]);

		/* ----------------------- Late code generation ----------------------- */
		/* Late adds +earlyLateSpc to Prompt. Keep the Early-to-Prompt and
		   Prompt-to-Late additions separate to preserve float32 rounding. */
		for (int k = 0; k < 2; ++k) {
			codePhase_reg1.val[k] = vaddq_f32(codePhase_reg1.val[k], earlyLateSpc_reg);
			codePhase_reg2.val[k] = vaddq_f32(codePhase_reg2.val[k], earlyLateSpc_reg);
			Index_reg1.val[k] = vcvtq_s32_f32(codePhase_reg1.val[k]);
			Index_reg2.val[k] = vcvtq_s32_f32(codePhase_reg2.val[k]);
		}
		EPL_reg = gatherCode16(caCode, Index_reg1, Index_reg2);
		vst1q_s16(lCode, EPL_reg.val[0]);
		vst1q_s16(lCode + 8, EPL_reg.val[1]);

		/* Move to the next batch of 16 code samples. */
		for (int k = 0; k < 2; ++k)
			stepNumber_reg.val[k] = vaddq_f32(stepNumber_reg.val[k], sixteenStep);
	}

	/* ------------ Scalar tail: remaining blksize % 16 samples ------------ */
	/* Recreate the double Early phase at the first leftover sample and
	   advance it by codePhaseStep. Index casts truncate toward zero; Late
	   uses Early + 2*earlyLateSpc directly. Chip values use scalar conversion
	   to short rather than the SIMD path's signed saturation. */
	double tempCodePhase = simdSamples * codePhaseStep + remCodePhase;
	double twoEarlyLateSpc = earlyLateSpc * 2;
	for (; sampleIndex < blksize; ++sampleIndex, ++eCode, ++pCode, ++lCode)
	{
		eCode[0] = caCode[(int)tempCodePhase];
		pCode[0] = caCode[(int)(tempCodePhase + earlyLateSpc)];
		lCode[0] = caCode[(int)(tempCodePhase + twoEarlyLateSpc)];
		tempCodePhase += codePhaseStep;
	}
}

/* Correlating function ------------------------------------------------------------
 * Correlate baseband samples with early/prompt/late local codes
 * Args :   short  *iBasebandSignal    I   I component of input signal with carrier wiped off
 *          short  *qBasebandSignal    I   Q component of input signal with carrier wiped off
 *          short  **localCode         I   local code sequences
 *          size_t blksize             I   number of baseband samples to correlate
 *          double *corrValues        O   outputs of correlator values arranged as [IE QE IP QP IL QL] 
 * Return : None
 *------------------------------------------------------------------------------*/
void correlator(short *iBasebandSignal, short *qBasebandSignal, short **localCode, size_t blksize, double *corrValues)
{
	const short* pI = iBasebandSignal;
	const short* pQ = qBasebandSignal;
	short* eCode = localCode[0];
	short* pCode = localCode[1];
	short* lCode = localCode[2];

	/* Multiply int16 baseband samples by int16 local-code chips and add
	   adjacent products: [a0*b0 + a1*b1, a2*b2 + a3*b3, ...]. Each 16-sample
	   block produces eight int32 partial sums, held in two NEON vectors
	   per output throughout the SIMD loop. */
	int16x8x2_t I_reg, Q_reg, EPL_reg;
	int32x4x2_t I_E_reg, I_P_reg, I_L_reg, Q_E_reg, Q_P_reg, Q_L_reg, mul_reg;
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
		/* ---------------------- Load baseband samples ---------------------- */
		/* Read two groups of eight int16 samples. Baseband buffers and the
		   three local-code buffers are allocated with 16-byte alignment. */
		for (int k = 0; k < 2; ++k) {
			I_reg.val[k] = vld1q_s16(pI + 8 * k);
			Q_reg.val[k] = vld1q_s16(pQ + 8 * k);
		}

		/* --------------- Compute the early I/Q correlation values --------------- */
		for (int k = 0; k < 2; ++k) {
			EPL_reg.val[k] = vld1q_s16(eCode + 8 * k);
			/* Widen int16 products to int32, then add adjacent pairs. */
			mul_reg.val[k] = vpaddq_s32(
				vmull_s16(vget_low_s16(I_reg.val[k]), vget_low_s16(EPL_reg.val[k])),
				vmull_s16(vget_high_s16(I_reg.val[k]), vget_high_s16(EPL_reg.val[k])));
			I_E_reg.val[k] = vaddq_s32(I_E_reg.val[k], mul_reg.val[k]);
			mul_reg.val[k] = vpaddq_s32(
				vmull_s16(vget_low_s16(Q_reg.val[k]), vget_low_s16(EPL_reg.val[k])),
				vmull_s16(vget_high_s16(Q_reg.val[k]), vget_high_s16(EPL_reg.val[k])));
			Q_E_reg.val[k] = vaddq_s32(Q_E_reg.val[k], mul_reg.val[k]);
		}

		/* --------------- Compute the prompt I/Q correlation values --------------- */
		for (int k = 0; k < 2; ++k) {
			EPL_reg.val[k] = vld1q_s16(pCode + 8 * k);
			/* Widen int16 products to int32, then add adjacent pairs. */
			mul_reg.val[k] = vpaddq_s32(
				vmull_s16(vget_low_s16(I_reg.val[k]), vget_low_s16(EPL_reg.val[k])),
				vmull_s16(vget_high_s16(I_reg.val[k]), vget_high_s16(EPL_reg.val[k])));
			I_P_reg.val[k] = vaddq_s32(I_P_reg.val[k], mul_reg.val[k]);
			mul_reg.val[k] = vpaddq_s32(
				vmull_s16(vget_low_s16(Q_reg.val[k]), vget_low_s16(EPL_reg.val[k])),
				vmull_s16(vget_high_s16(Q_reg.val[k]), vget_high_s16(EPL_reg.val[k])));
			Q_P_reg.val[k] = vaddq_s32(Q_P_reg.val[k], mul_reg.val[k]);
		}

		/* ---------------- Compute the late I/Q correlation values ---------------- */
		for (int k = 0; k < 2; ++k) {
			EPL_reg.val[k] = vld1q_s16(lCode + 8 * k);
			/* Widen int16 products to int32, then add adjacent pairs. */
			mul_reg.val[k] = vpaddq_s32(
				vmull_s16(vget_low_s16(I_reg.val[k]), vget_low_s16(EPL_reg.val[k])),
				vmull_s16(vget_high_s16(I_reg.val[k]), vget_high_s16(EPL_reg.val[k])));
			I_L_reg.val[k] = vaddq_s32(I_L_reg.val[k], mul_reg.val[k]);
			mul_reg.val[k] = vpaddq_s32(
				vmull_s16(vget_low_s16(Q_reg.val[k]), vget_low_s16(EPL_reg.val[k])),
				vmull_s16(vget_high_s16(Q_reg.val[k]), vget_high_s16(EPL_reg.val[k])));
			Q_L_reg.val[k] = vaddq_s32(Q_L_reg.val[k], mul_reg.val[k]);
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

	/* ----------------- Scalar tail: remaining blksize % 16 samples ----------------- */
	/* Accumulate the remaining 0..15 samples with int64 products and sums. */
	for (; sampleIndex < blksize; ++sampleIndex, ++pI, ++pQ, ++eCode, ++pCode, ++lCode) {
		I_E += (long long)pI[0] * eCode[0];
		Q_E += (long long)pQ[0] * eCode[0];
		I_P += (long long)pI[0] * pCode[0];
		Q_P += (long long)pQ[0] * pCode[0];
		I_L += (long long)pI[0] * lCode[0];
		Q_L += (long long)pQ[0] * lCode[0];
	}
	/* Convert to double and apply the nominal 1/16 carrier-LUT normalization.
	   This does not reverse the caller's input right shift.
	   Output order is [IE QE IP QP IL QL]. */
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
 * Return : void*                        aligned pointer, or NULL on allocation failure
 *------------------------------------------------------------------------------*/
static void* checkedAlignedMalloc(size_t nbytes, const char* name)
{
	void* p = NULL;
	if (posix_memalign(&p, 16, nbytes) != 0) {
		p = NULL;
		printf("   Aligned memory allocation failed for %s.\n", name);
	}
	return p;
}

/* gatherCode16 ----------------------------------------------------------------
 * Gather 16 int32 code chips and narrow them with signed saturation
 * into two NEON vectors of 8 x int16 samples. All indexes must refer to
 * valid elements of caCode; this helper does not check bounds.
 * Args   : const int* caCode         I   int32 code table for one PRN
 *          int32x4x2_t Index_reg1   I   code indexes for samples 0..7
 *          int32x4x2_t Index_reg2   I   code indexes for samples 8..15
 * Return : int16x8x2_t                  gathered code chips in chronological order
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
		printf("   Invalid input arguments for corrNeonSIMDSerialBPSK.\n");
		return 0;
	}
	return 1;
}

/* cleanup --------------------------------------------------------------------
 * Free the reusable work buffers and reset the initialization flag.
 * The next corrEngine() call allocates fresh buffers. Safe to call repeatedly.
 * Return : None
 *------------------------------------------------------------------------------*/
void cleanup(void)
{
	printf("   SIMD DLL is terminating, destroying allocated memory ...\n");

	/* --------------- Free memory --------------- */
	for (int i = 0; i < 3; ++i) {
		free(localCode[i]);
		localCode[i] = NULL;
	}

	free(iBasebandSignal); iBasebandSignal = NULL;
	free(qBasebandSignal); qBasebandSignal = NULL;
	initialized = 0;
}
