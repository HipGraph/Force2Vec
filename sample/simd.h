/*
 * NOTE: The purpose of this header file is to hide all architecture 
 * dependent implementation of intrinsic vector codes 
 * NOTE: some parts of the codes are inspired and modified from atlas_simd.h of 
 * ATLAS: https://github.com/math-atlas/math-atlas  
 */
#ifndef _SIMD_H_
#define _SIMD_H_

#define BLC_X86   
#define BLC_AVXZ 
/*#define BLC_AVX2 */
 /*
  *   inst format: inst(dist, src1, src2)
  */
#ifdef BLC_X86
   #if defined(BLC_AVXZ) || defined(BLC_AVX512) /* avx512f */
      #include<immintrin.h>
      #define VLENb 64
      #if defined(DREAL)
         #define VLEN 8
/*
 *       AVX512 double precision 
 */
         #define VTYPE __m512d 
         #define BCL_vldu(v_, p_) v_ = _mm512_loadu_pd(p_) 
         #define BCL_vld(v_, p_) v_ = _mm512_load_pd(p_) 
         #define BCL_vzero(v_) v_ = _mm512_setzero_pd() 
         #define BCL_vstu(p_, v_) _mm512_storeu_pd(p_, v_) 
         #define BCL_vst(p_, v_)  _mm512_store_pd(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm512_set1_pd(*(p_))
         #define BCL_vset1(v_, f_) v_ = _mm512_set1_pd(f_)
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm512_add_pd(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm512_sub_pd(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm512_mul_pd(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm512_div_pd(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = _mm512_fmadd_pd(s1_, s2_, d_)
         #define BCL_vmax(d_, s1_, s2_) d_ = _mm512_max_pd(s1_, s2_)
         #define BCL_vmin(d_, s1_, s2_) d_ = _mm512_min_pd(s1_, s2_)
         #define BCL_vrcp(d_, s1_) d_ = _mm512_rcp14_pd(s1_); // reciprocal 
         #define BCL_maskz_vrcp(d_, k_, s_) d_ = _mm512_rcp14_pd(k_, s_); // reciprocal 
         #define BCL_imaskz_vrcp(d_, ik_, s_) \
         {  __mmask8 k0_ = _cvtu32_mask8(ik_); \
            d_ = _mm512_maskz_rcp14_pd(k0_, s_);\
         }
         #define BCL_cvtint2mask(k_, ik) k_ = _cvtu32_mask8(ik_) 
/*
 *       VVRSUM codes from ATLAS 
 */
         /* vector to scalar */
         #define BCL_vrsum1(d_, s_) \
         { __m256d t0_, t1_; __m128d x0_, x1_; \
            t0_ = _mm512_extractf64x4_pd(s_, 0); \
            t1_ = _mm512_extractf64x4_pd(s_, 1); \
            t0_ = _mm256_add_pd(t0_, t1_); \
            x0_ = _mm256_extractf128_pd(t0_, 0); \
            x1_ = _mm256_extractf128_pd(t0_, 1); \
            x0_ = _mm_add_pd(x0_, x1_); \
            x0_ = _mm_hadd_pd(x0_, x0_); \
            d_ = x0_[0];  \
         }
      #else  /* elif defined (SREAL) */
         #define VLEN 16
         #define VTYPE __m512 
         #define BCL_cvtint2mask(ik_, k_) k_ = _cvtu32_mask8(ik_) 
         #define BCL_vldu(v_, p_) v_ = _mm512_loadu_ps(p_) 
         #define BCL_vld(v_, p_) v_ = _mm512_load_ps(p_) 
         #define BCL_vzero(v_) v_ = _mm512_setzero_ps() 
         #define BCL_vstu(p_, v_) _mm512_storeu_ps(p_, v_) 
         #define BCL_vst(p_, v_)  _mm512_store_ps(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm512_set1_ps(*(p_))
         #define BCL_vset1(v_, f_) v_ = _mm512_set1_ps(f_)
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm512_add_ps(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm512_sub_ps(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm512_mul_ps(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm512_div_ps(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = _mm512_fmadd_ps(s1_, s2_, d_)
         #define BCL_vmax(d_, s1_, s2_) d_ = _mm512_max_ps(s1_, s2_)
         #define BCL_vmin(d_, s1_, s2_) d_ = _mm512_min_ps(s1_, s2_)
         #define BCL_vrcp(d_, s1_) d_ = _mm512_rcp14_ps(s1_); /* reciprocal */
         #define BCL_maskz_vrcp(d_, k_, s_) d_ = _mm512_maskz_rcp14_ps(k_, s_); 
         #define BCL_imaskz_vrcp(d_, ik_, s_) \
         {  __mmask8 k0_ = _cvtu32_mask8(ik_); \
            d_ = _mm512_maskz_rcp14_ps(k0_, s_);\
         }
         
         /* vector reduced to a variable: from ATLAS */
         #define BCL_vrsum1(d_, s_) \
         { __m256 t0_, t1_; __m128 x0_, x1_; \
            t0_ = _mm512_extractf32x8_ps(s_, 0); \
            t1_ = _mm512_extractf32x8_ps(s_, 1); \
            t0_ = _mm256_add_ps(t0_, t1_); \
            x0_ = _mm256_extractf128_ps(t0_, 0); \
            x1_ = _mm256_extractf128_ps(t0_, 1); \
            x0_ = _mm_add_ps(x0_, x1_); \
            x0_ = _mm_hadd_ps(x0_, x0_); /* {X,X,x0dc,x0ab} */ \
            x0_ = _mm_hadd_ps(x0_, x0_); /* {X,X,X,x0abcd} */ \
            d_ = x0_[0]; \
         }
      #endif
/*
 * AVX 
 */
   #elif defined(BLC_AVX2) || defined(BLC_AVXMAC) || defined(BLC_AVX) 
      #include<immintrin.h>
      #define VLENb 32
      #if defined(BLC_AVX2) || defined(BLC_AVXMAC)
         #define ArchHasMAC 
      #endif
      #if defined (DREAL)
         #define VLEN 4
         #define VTYPE __m256d 
         #define BCL_vldu(v_, p_) v_ = _mm256_loadu_pd(p_) 
         #define BCL_vld(v_, p_) v_ = _mm256_load_pd(p_) 
         #define BCL_vzero(v_) v_ = _mm256_setzero_pd() 
         #define BCL_vstu(p_, v_) _mm256_storeu_pd(p_, v_) 
         #define BCL_vst(p_, v_)  _mm256_store_pd(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm256_set1_pd(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm256_add_pd(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm256_sub_pd(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm256_mul_pd(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm256_div_pd(s1_, s2_)
         #ifdef ArchHasMAC
            #define BCL_vmac(d_, s1_, s2_) d_ = _mm256_fmadd_pd(s1_, s2_, d_)
         #else
            #define BCL_vmac(d_, s1_, s2_) \
            {  VTYPE vt_; \
               vt_ = _mm256_mul_pd(s1_, s2_); \
               d_ = _mm256_add_pd(vt_, d_); \
            }
         #endif
         /* NOTE: no reciprocal for double precision, only for single precision
          *       So, we use multiple ints to make it work... 
          */
         #define BCL_vrcp(d_, s_) \
         {   VTYPE _vx = _mm256_set1_pd(1.0); \
             d_ = _mm256_div_pd(_vx, s_); \
         }
         /* FIXME: need to test: src always read-only  */
         #define BCL_imaskz_vrcp(d_, ik_ s_) \
         {  VTYPE v0_ = _mm256_setzero_pd();\
            VTYPE v1_ = _mm256_set1_pd(1.0);\
            d_ = _mm256_blend_pd(s_, v1_, ik_); \
            d_ = _mm256_div_pd(v1_, d_); \
            d_ = _mm256_blend_pd(d_, v0_, ik_); \
         }
         /*#define BCL_cvtint2mask(k_, ik) k_ = _cvtu32_mask8(ik_) */
      #else /* SREAL */
         #define VLEN 8
         #define VTYPE __m256 
         #define BCL_vldu(v_, p_) v_ = _mm256_loadu_ps(p_) 
         #define BCL_vld(v_, p_) v_ = _mm256_load_ps(p_) 
         #define BCL_vzero(v_) v_ = _mm256_setzero_ps() 
         #define BCL_vstu(p_, v_) _mm256_storeu_ps(p_, v_) 
         #define BCL_vst(p_, v_)  _mm256_store_ps(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm256_set1_ps(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm256_add_ps(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm256_sub_ps(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm256_mul_ps(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm256_div_ps(s1_, s2_)
         #ifdef ArchHasMAC
            #define BCL_vmac(d_, s1_, s2_) d_ = _mm256_fmadd_ps(s1_, s2_, d_)
         #else
            #define BCL_vmac(d_, s1_, s2_) \
            {  VTYPE vt_; \
               vt_ = _mm256_mul_ps(s1_, s2_); \
               d_ = _mm256_add_ps(vt_, d_); \
            }
         #endif
         /* NOTE: no reciprocal for double precision, only for single precision */
         #if 1
            #define BCL_vrcp(d_, s_) d_ = _mm256_rcp_ps(s_); /* reciprocal */
         #else
            #define BCL_vrcp(d_, s_) \
            {  VTYPE _vx = _mm256_set1_ps(1.0); \
               d_ = _mm256_div_ps(_vx, s_); \
            }
         #endif
         /* #define BCL_maskz_vrcp(k_, d_) d_ = _mm256_rcp14_ps(k_, d_); // reciprocal */
         /*  FIXME: need to test */
         #if 1
            #define BCL_imaskz_vrcp(d_, ik_, s_) \
            {  VTYPE v0_ = _mm256_setzero_ps();\
               VTYPE v1_ = _mm256_set1_ps(1.0);\
               d_ = _mm256_blend_ps(s_, v1_, ik_); \
               d_ = _mm256_rcp_ps(d_); \
               d_ = _mm256_blend_ps(d_, v0_, ik_); \
            }
         #else
            #define BCL_imaskz_vrcp(d_, ik_, s_) \
            {  VTYPE v0_ = _mm256_setzero_ps();\
               VTYPE v1_ = _mm256_setzero+ps(1.0);\
               d_ = _mm256_blend_ps(s_, v1_, ik_); \
               d_ = _mm256_div_ps(v1_, d_); \
               d_ = _mm256_blend_ps(d_, v0_, ik_); \
            }
         #endif
      #endif
   #elif defined(BLC_SSE2) || defined(BLC_SSE3) || defined(BLC_SSE4.1) \
         || defined(BLC_SSE4.2)
      #define VLENb 16
      #if defined(DREAL)
         #define VLEN 2
         #define VTYPE __m128d 
         #define BCL_vldu(v_, p_) v_ = _mm_loadu_pd(p_) 
         #define BCL_vld(v_, p_) v_ = _mm_load_pd(p_) 
         #define BCL_vzero(v_) v_ = _mm_setzero_pd() 
         #define BCL_vstu(p_, v_) _mm_storeu_pd(p_, v_) 
         #define BCL_vst(p_, v_)  _mm_store_pd(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm_set1_pd(*(p_))
         #define BCL_vset1(v_, f_) v_ = _mm_set1_pd(f_)
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm_add_pd(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm_sub_pd(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm_mul_pd(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm_div_pd(s1_, s2_)
         
         /*Note: no MAC support for SSE */
         #define BCL_vmac(d_, s1_, s2_) \
         {  VTYPE vt_; \
            vt_ = _mm_mul_pd(s1_, s2_); \
            d_ = _mm_add_pd(vt_, d_); \
         }
         #define BCL_vmax(d_, s1_, s2_) d_ = _mm_max_pd(s1_, s2_)
         #define BCL_vmin(d_, s1_, s2_) d_ = _mm_min_pd(s1_, s2_)
         
         /* no direct reciprocal support on SSE */
         #define BCL_vrcp(d_, s_) \
         {  VTYPE _vx = _mm_set1_pd(1.0); \
            d_ = _mm_div_pd(_vx, s_); \
         }
         #if defined(SSE4.1) || defined(SSE4.2)
            /* NOTE: ik_ must be const int imm8  */ 
            #define BCL_imaskz_vrcp(d_, ik_, s_) \
            {  VTYPE v0_ = _mm_setzero_pd();\
               VTYPE v1_ = _mm_set1_pd(1.0);\
               d_ = _mm_blend_pd(s_, v1_, ik_); \
               d_ = _mm_div_pd(v1_, d_); \
               d_ = _mm_blend_pd(d_, v0_, ik_); \
            }
         #else
            #error "BCL_imaskz_vrcp not supported prior to SSE4.1!"
         #endif
         /*
          * NOTE: other form of MASK inst not supported 
          */
/*
 *       VVRSUM codes from ATLAS 
 */
         /* vector to scalar */
         #if defined(SSE3) || defined(SSE4.1) || defined(SSE4.2)
            #define BCL_vrsum1(d_, s_) d_ = _mm_cvtsd_f64(_mm_hadd_pd(s_, s_))
         #else /* SSE2, hadd not supported */
            #define BCL_vrsum1(d_, s_) \
               d_ = _mm_cvtsd_f64(_mm_add_sd(_mm_unpackhi_pd(s_, s_), s_))
         #endif
      #else /* single precision float */
         #define VLEN 4
         #define VTYPE __m128 
         #define BCL_vldu(v_, p_) v_ = _mm_loadu_ps(p_) 
         #define BCL_vld(v_, p_) v_ = _mm_load_ps(p_) 
         #define BCL_vzero(v_) v_ = _mm_setzero_ps() 
         #define BCL_vstu(p_, v_) _mm_storeu_ps(p_, v_) 
         #define BCL_vst(p_, v_)  _mm_store_ps(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm_set1_ps(*(p_))
         #define BCL_vset1(v_, f_) v_ = _mm_set1_ps(f_)
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm_add_ps(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm_sub_ps(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm_mul_ps(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm_div_ps(s1_, s2_)
         /*Note: no MAC support for SSE */
         #define BCL_vmac(d_, s1_, s2_) \
         {  VTYPE vt_; \
            vt_ = _mm_mul_ps(s1_, s2_); \
            d_ = _mm_add_ps(vt_, d_); \
         }
         #define BCL_vmax(d_, s1_, s2_) d_ = _mm_max_ps(s1_, s2_)
         #define BCL_vmin(d_, s1_, s2_) d_ = _mm_min_ps(s1_, s2_)
         
         /* no direct reciprocal support on SSE */
         #define BCL_vrcp(d_, s_) \
         {  VTYPE _vx = _mm_set1_ps(1.0); \
            d_ = _mm_div_ps(_vx, s_); \
         }
         /* blend operation supported from SSE4.1 */
         #if defined(SSE4.1) || defined(SSE4.2)
            /* NOTE: ik_ must be const int imm8  */ 
            #define BCL_imaskz_vrcp(d_, ik_, s_) \
            {  VTYPE v0_ = _mm_setzero_ps();\
               VTYPE v1_ = _mm_set1_ps(1.0);\
               d_ = _mm_blend_ps(s_, v1_, ik_); \
               d_ = _mm_div_ps(v1_, d_); \
               d_ = _mm_blend_ps(d_, v0_, ik_); \
            }
         #else
            #error "BCL_imaskz_vrcp not supported prior to SSE4.1!"
         #endif
/*
 *       VVRSUM codes from ATLAS 
 */
         #if defined(SSE3) || defined(SSE4.1) || defined(SSE4.2)
            #define BCL_vrsum1(d_, s_) \
            {  VTYPE t_; \
               t_ = _mm_hadd_ps(s_, s_); \
               d_ = _mm_cvtss_f32(_mm_hadd_ps(t_, t_)); \
            }
         #else /* SSE2 */
            #define BCL_vrsum1(d_, s_) \
            { \
               VTYPE t_; \
               t_ = _mm_movehl_ps(s_, s_); \
               t_ = _mm_add_ps(t_, s_); \
               t_ = _mm_add_ps(t_, _mm_shuffle_ps(t_, t_, 1)); \
               d_ = _mm_cvtss_f32(t_); \
            }
         #endif
      #endif
   #elif defined(BLC_SSE1)
      #define VLENb 16
      #if defined(SREAL)
         #define VLEN 4
         #define VTYPE __m128 
         #define BCL_vldu(v_, p_) v_ = _mm_loadu_ps(p_) 
         #define BCL_vld(v_, p_) v_ = _mm_load_ps(p_) 
         #define BCL_vzero(v_) v_ = _mm_setzero_ps() 
         #define BCL_vstu(p_, v_) _mm_storeu_ps(p_, v_) 
         #define BCL_vst(p_, v_)  _mm_store_ps(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm_set1_ps(*(p_))
         #define BCL_vset1(v_, f_) v_ = _mm_set1_ps(f_)
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm_add_ps(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm_sub_ps(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm_mul_ps(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm_div_ps(s1_, s2_)
         /*Note: no MAC support for SSE */
         #define BCL_vmac(d_, s1_, s2_) \
         {  VTYPE vt_; \
            vt_ = _mm_mul_ps(s1_, s2_); \
            d_ = _mm_add_ps(vt_, d_); \
         }
         #define BCL_vmax(d_, s1_, s2_) d_ = _mm_max_ps(s1_, s2_)
         #define BCL_vmin(d_, s1_, s2_) d_ = _mm_min_ps(s1_, s2_)
         
         /* no direct reciprocal support on SSE */
         #define BCL_vrcp(d_, s_) \
         {  VTYPE _vx = _mm_set1_ps(1.0); \
            d_ = _mm_div_ps(_vx, s_); \
         }
/*
 *       VVRSUM codes from ATLAS 
 */
         #define BCL_vrsum1(d_, s_) \
         { \
               VTYPE t_; \
               t_ = _mm_movehl_ps(s_, s_); \
               t_ = _mm_add_ps(t_, s_); \
               t_ = _mm_add_ps(t_, _mm_shuffle_ps(t_, t_, 1)); \
               d_ = _mm_cvtss_f32(t_); \
         }
      #else /* double not supported */
         #error "double type not supported on SSE1 !"
      #endif
   #else
      #error "Unsupported X86 SIMD!"
   #endif

#elif defined(BLC_VSX)  /* openPower vector unit */
   #include <altivec.h>   
   #define VLENb 16
   #if defined(DREAL)
      #define VLEN 2
      #define VTYPE vector double  
   #else /* single precision float */
      #define VLEN 4
      #define VTYPE vector float 
   #endif
   #define BLC_vldu(v_, p_) v_ = vec_vsx_ld(0, (VTYPE*)(p_)) 
   #define BLC_vld(v_, p_) v_ = vec_ld(0, (VTYPE*)(p_))  
   #define BLC_vzero(v_) v_ = vec_splats((VALUETYPE)0.0)
   #define BLC_vstu(p_, v_) vec_vsx_st(v_, 0, (VTYPE*)(p_))
   #define BLC_vst(p_, v_)  vec_st(v_, 0, (VTYPE*)(p_))
   #define BLC_vbcast(v_, p_) v_ =  vec_splats(*((VALUETYPE*)(p_)))
   #define BLC_vadd(d_, s1_, s2_) d_ =  vec_add(s1_, s2_) 
   #define BLC_vsub(d_, s1_, s2_) d_ =  vec_sub(s1_, s2_) 
   #define BLC_vmul(d_, s1_, s2_) d_ =  vec_mul(s1_, s2_) 
   #define BLC_vdiv(d_, s1_, s2_) d_ =  vec_div(s1_, s2_) 
   #define BLC_vmac(d_, s1_, s2_) d_ =  vec_madd(s1_, s2_, d_) 
   #define BLC_vmax(d_, s1_, s2_) d_ =  vec_max(s1_, s2_) 
   #define BLC_vmin(d_, s1_, s2_) d_ =  vec_min(s1_, s2_) 
   #define BCL_vrcp(d_, s_) d_ = vec_re(s_); /* reciprocal */
   /* FIXME: need to use vec_se to implement masked rcp  
   //#define BCL_imaskz_vrcp(d_, ik_) \ */
   #ifdef DREAL 
      #define BCL_vrsum1(d_, s_) \
      { \
         d_ = vec_splat(s_, 1); \
         d_ = vec_add(d_, s_) ; \
      }
   #else
      #define BCL_vrsum1(d_, s_) \
      { \
         VTYPE t_; \
         d_ = vec_splat(s_, 1); \
         d_ = vec_add(d_, s_) ; \
         t_ = vec_splat(s_, 2); \
         d_ = vec_add(d_, t_) ; \
         t_ = vec_splat(s_, 3); \
         d_ = vec_add(d_, t_) ; \
      }
   #endif
#elif defined(BLC_ARM64) /* arm64 machine */
   #define VLENb 16
   #include "arm_neon.h"
   #if defined(BCL_GAS_ARM) && defined(DREAL)
         #define VLEN 2
         #define VTYPE float64x2_t
         #define BCL_vzero(v_) v_ = vdupq_n_f64(0.0)
         #define BCL_vbcast(v_, p_) v_ =  vld1q_dup_f64(p_)
         #define BCL_vld(v_, p_) v_ = vld1q_f64(p_)
         #define BCL_vst(p_, v_) vst1q_f64(p_, v_)
         #define BCL_vadd(d_, s1_, s2_) d_ = vaddq_f64(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = vsubq_f64(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = vmulq_f64(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = vfmaq_f64(d_, s1_, s2_)
         #define BCL_vmax(d_, s1_, s2_) d_ = vmax_f64(d_, s1_, s2_)
         #define BCL_vmin(d_, s1_, s2_) d_ = vmin_f64(d_, s1_, s2_)
         #define BCL_vrcp(d_, s_) d_ = vrecps_f64(s_)
         #define BCL_vrsum1(d_, s_) d_ = vget_low_f64(vpaddq_f64(s_, s_))
   #elif defined(BCL_NEON) || defined(SREAL) /* NEON SIMD unit */
         #define VLEN 4
         #define VTYPE float32x4_t
         #define BCL_vzero(v_) v_ = vdupq_n_f32(0.0f)
         #define BCL_vbcast(v_, p_) v_ =  vdupq_n_f32(*(p_));
         #define BCL_vset1(v_, f_) v_ = vdupq_n_f32(f);
         #define BCL_vld(v_, p_) v_ = vld1q_f32(p_)
         #define BCL_vst(p_, v_) vst1q_f32(p_, v_)
         #define BCL_vadd(d_, s1_, s2_) d_ = vaddq_f32(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = vsubq_f32(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = vmulq_f32(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = vmlaq_f32(d_, s1_, s2_)
         #define BCL_vmax(d_, s1_, s2_) d_ = vmax_f32(d_, s1_, s2_)
         #define BCL_vmin(d_, s1_, s2_) d_ = vmin_f32(d_, s1_, s2_)
         #define BCL_vrcp(d_, s_) d_ = vrecps_f32(s_)
         #define BCL_vrsum1(d_, s_) \
         {  VTYPE t4_; float32x2_t t2_, t1_; \
            t1_ = vget_high_f32(s_); \
            t2_ = vget_low_f32(s_); \
            t2_ = vpadd_f32(t1_, t2_); \
            d_ = vget_lane_f32(t2_, 0); \
            d_ += vget_lane_f32(t2_, 1); \
         }
   #else
      #error "Unsupported ARM SIMD !"
   #endif
#elif defined(BLC_FRCGNUVEC) /* GNUVEC by GCC  */
   #error "Unsupported Architecture!"
#endif

/*
 * If not defined, run slow version which should work with any gcc-compatible
 * compiler. [from ATLAS's atlas_simd.h file]
 */
#ifndef BCL_vrsum1
   #define BCL_vrsum1(d_, s_) \
   {  TYPE mem_[VLEN] __attribute__ ((aligned (VLENb)));\
      int i_; \
      BCL_vst(mem_, s_); \
      d_ = *mem_; \
      for (i_=1; i_ < VLEN; i_++) \
         d_ += mem_[i_]; \
   }
#endif

#endif
