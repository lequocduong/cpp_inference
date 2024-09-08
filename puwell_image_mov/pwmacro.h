/*----------------------------------------------------------------------------------------------
*
* This file is Puwell's property. It contains Puwell's trade secret, proprietary and 		
* confidential information. 
* 
* The information and code contained in this file is only for authorized Puwell employees 
* to design, create, modify, or review.
* 
* DO NOT DISTRIBUTE, DO NOT DUPLICATE OR TRANSMIT IN ANY FORM WITHOUT PROPER AUTHORIZATION.
* 
* If you are not an intended recipient of this file, you must not copy, distribute, modify, 
* or take any action in reliance on it. 
* 
* If you have received this file in error, please immediately notify Puwell and 
* permanently delete the original and any copy of any file and any printout thereof.
*
*---------------------------------------------------------------------------------------------*/

#ifndef __PWMACRO_H__
#define __PWMACRO_H__

#define PW(fun)			lgy##fun
		
//	#define GrowMask		PW(GrowMask)

#ifndef	PI
#define PI				3.1415926535
#define PIF				3.14169265f
#endif
#ifndef	PI_20
#define PI_20			3294199
#endif

#define GNull		0
#define GFalse		0
#define GTrue		1

#ifndef PW_MAX_PATH
#define PW_MAX_PATH	256
#endif

#ifndef	PW_FABS
#define PW_FABS(x)		(((x) < 0) ? -(x) : (x))
#endif

#ifndef	PWABS8
#define PWABS8(x)		(((x)+((x)>>7))^((x)>>7))
#endif

#ifndef	PWABS16
#define PWABS16(x)		(((x)+((x)>>15))^((x)>>15))
#endif

#ifndef	PWABS32
#define PWABS32(x)		(((x)+((x)>>31))^((x)>>31))
#endif

#ifndef	PWABS64
#define PWABS64(x)		(((x)+((x)>>63))^((x)>>63))
#endif


#ifndef	PW_MAX
#define PW_MAX(x,y)		(((x)>=(y))?(x):(y))
#endif

#ifndef	PW_MIN
#define PW_MIN(x,y)		(((x)<=(y))?(x):(y))
#endif

#ifndef	PW_SWAP
#define PW_SWAP(x,y,t)	((t) = (x), (x) = (y), (y) = (t))
#endif

#ifndef	PW_FPI2D
#define PW_FPI2D(x)	((x)*57.29578f)  // x*180/PI
#endif

#ifndef	PW_FD2PI
#define PW_FD2PI(x)	((x)*0.01745329f)  // x*PI/180
#endif

#ifndef	PW_PI2D_10
#define PW_PI2D_10(x)	((x)*58671)  // x*180/PI
#endif

#ifndef	PW_D2PI_10
#define PW_D2PI_10(x)	(((x)*143)>>3)  // x*PI/180
#endif

#define PW_ROUND(x)		((GInt32)((x)+0.5))
#define PW_ROUND_S(x)	( (x) > 0 ? (GInt32)((x)+0.5) : (GInt32)((x)-0.5) )
#define PW_ROUND_N(x, type, times)		(type)((x)*times+0.5)

#define PW_ROUNDF(x)		((GInt32)((x)+0.5f))
#define PW_ROUND_SF(x)	( (x) > 0 ? (GInt32)((x)+0.5f) : (GInt32)((x)-0.5f) )
#define PW_ROUND_NF(x, type, times)		(type)((x)*times+0.5f)

#define PW_TRIMBYTE(x)	(GUInt8)((x)&(~255)?((-(x))>>31):(x))
#define PW_TRIM16(x)	(GUInt16)((x)&(~65535)?((-(x))>>31):(x))


#define	yuv_shift		14
#define	yuv_fix(x)		(GInt32)((x) * (1 << (yuv_shift)) + 0.5f)
#define	yuv_descale(x)	(((x) + (1 << ((yuv_shift)-1))) >> (yuv_shift))
#define	yuv_prescale(x)	((x) << yuv_shift)

#define	yuvYr	yuv_fix(0.299f)
#define	yuvYg	yuv_fix(0.587f)
#define	yuvYb	yuv_fix(0.114f)
#define	yuvCr	yuv_fix(0.713f)
#define	yuvCb	yuv_fix(0.564f)

#define	yuvRCr	yuv_fix(1.403f)
#define	yuvGCr	(-yuv_fix(0.714f))
#define	yuvGCb	(-yuv_fix(0.344f))
#define	yuvBCb	yuv_fix(1.773f)

#define	ET_CAST_8U(t)		(GUInt8)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)

#define ET_YUV_TO_R(y,v)	(GUInt8)(ET_CAST_8U(yuv_descale((yuv_prescale(y)) + yuvRCr * (v))))
#define ET_YUV_TO_G(y,u,v)	(GUInt8)(ET_CAST_8U(yuv_descale((yuv_prescale(y)) + yuvGCr * (v) + yuvGCb * (u))))
#define ET_YUV_TO_B(y,u)	(GUInt8)(ET_CAST_8U(yuv_descale((yuv_prescale(y)) + yuvBCb * (u))))

// #define ET_RGB_TO_Y(r,g,b)	(GLong)(yuv_descale((b) * yuvYb + (g) * yuvYg + (r) * yuvYr))
// #define ET_RGB_TO_U(y,b)	(GLong)(yuv_descale(((b) - (y)) * yuvCb) + 128)
// #define ET_RGB_TO_V(y,r)	(GLong)(yuv_descale(((r) - (y)) * yuvCr) + 128)
#define ET_RGB_TO_Y(r,g,b)	(GUInt8)(ET_CAST_8U(yuv_descale((b) * yuvYb + (g) * yuvYg + (r) * yuvYr)))
#define ET_RGB_TO_U(y,b)	(GUInt8)(ET_CAST_8U(yuv_descale(((b) - (y)) * yuvCb) + 128))
#define ET_RGB_TO_V(y,r)	(GUInt8)(ET_CAST_8U(yuv_descale(((r) - (y)) * yuvCr) + 128))

#define ET_RGB_TO_US(y,b)	(GInt32)(yuv_descale(((b) - (y)) * yuvCb))
#define ET_RGB_TO_VS(y,r)	(GInt32)(yuv_descale(((r) - (y)) * yuvCr))

#define AILLINE_BYTES(w, b)	(((GInt32)(w) * (b) + 31) / 32 * 4)

#define PW_SET1_4(t)	(t)
#define PW_SET2_4(t)	((t)<<8)
#define PW_SET3_4(t)	((t)<<16)
#define PW_SET4_4(t)	((t)<<24)

#define PW_GET1_4(t)	((t) & 0x0ff)
#define PW_GET2_4(t)	(((t)<<16)>>24)
#define PW_GET3_4(t)	(((t)<<8)>>24)
#define PW_GET4_4(t)	((t)>>24)


#define PA_PROTOCOL_8	   		0x78	
#define PA_PROTOCOL_9      		0x79	
#define PA_PROTOCOL_PRODUCT		0x7a	

#define PW_CHECK_PARAMETER( EXT, Src, IllegalValue )		{	if( (IllegalValue) == (Src) ) goto EXT;	}
#define PW_CHECK_PARAMETER_NOT( EXT, Src, legalValue )		{	if( (legalValue) != (Src) ) goto EXT;	}
#define PW_CHECK_PARA_VAL( EXT, Src, IllegalValue, Res, Val )	{	if( (IllegalValue) == (Src) ) {Res=Val;goto EXT;}	}
#define PW_CHECK_PARA_VAL_NOT( EXT, Src, legalValue, Res, Val )	{	if( (legalValue) != (Src) ) {Res=Val;goto EXT;}	}

#define PW_CHK		PW_CHECK_PARAMETER
#define PW_CHK_N	PW_CHECK_PARAMETER_NOT
#define PW_CHK_V	PW_CHECK_PARA_VAL
#define PW_CHK_V_N	PW_CHECK_PARA_VAL_NOT


#define GO(res, EXIT, FUNCTION, OK)		{if((res = FUNCTION) != (OK)) goto EXIT;} 
#define UNGO(res, EXIT, FUNCTION, NOK)		{if((res = FUNCTION) == (NOK)) goto EXIT;} 
//	GO(res, EXIT, begood, GOK)
#define GOF(EXIT, FUNCTION, OK)	{if((FUNCTION) != (OK)) goto EXIT;} 
#define UNGOF(EXIT, FUNCTION, NOK)	{if((FUNCTION) == (NOK)) goto EXIT;} 

#define GOFV(EXIT, FUNCTION, OK, res, REV)	{if((FUNCTION) != (OK)) {(res)=(REV);goto EXIT;}} 
#define UNGOFV(EXIT, FUNCTION, NOK, res, REV)	{if((FUNCTION) == (NOK))  {(res)=(REV);goto EXIT;}} 

#define GOE(res, ERR, EXIT)			{res = ERR; goto EXIT;} 
//	GOE(res, GERR_NO_MEMORY, EXT);

#ifdef _WIN32
#define PW_STRCAT( strD, len, strS ) {strcat_s( strD, len, strS );}
#else
#define PW_STRCAT( strD, len, strS ) {                             \
        if( len > 0 ){                                             \
        GInt32 lS = strlen(strS), lD = strlen(strD);               \
        strncat( (strD), (strS), PW_MAX( (len)-1-lD, 0 ) );        \
        (strD)[PW_MIN(lS+lD,(len)-1)]=0; } }
#endif

#ifdef _WIN32
#define PW_STRCPY( strD, len, strS ) {strcpy_s( strD, len, strS );}
#else
// #define PW_STRCPY( strD, len, strS ) {	\
// 	int _t = 0;							\
// 	char *_pS = strS, *_pD = strD;		\
// 	while( _t < len-1 && *_pS ){		\
// 		*_pD = *_pS;	_pS++; _pD++;	\
// 	}									\
// 	*_pD = 0;							\
// }
#define PW_STRCPY( strD, len, strS ) {                \
        if( len > 0 ){                                \
        GInt32 lS = strlen(strS);                     \
        strncpy( (strD), (strS), (len)-1 );           \
        (strD)[PW_MIN( lS,(len)-1 )]=0;}}
#endif

// #define PW_STRCPY( strD, len, strS ) {		\
// 	if( len > 0 ){							\
// 	GInt32 lS = strlen(strS);				\
// 	strncpy( (strD), (strS), (len)-1 );		\
// 	strD[PW_MIN( lS,(len)-1 )]=0;}}
// 
// #define PW_STRCAT( strD, len, strS ) {					\
// 	if( len > 0 ){										\
// 	GInt32 lS = strlen(strS), lD = strlen(strD);		\
// 	strncat( (strD), (strS), PW_MAX( (len)-1-lD, 0 ) );	\
// 	strD[PW_MIN(lS+lD,(len)-1)]=0; } }

// #ifdef _WIN32
// #define PW_STRCPY( strD, len, strS ) {		\
// 	if( len > 0 ){							\
// 	GInt32 lS = strlen(strS);				\
// 	strcpy_s( (strD), (len)-1, (strS) );	\
// 	(strD)[PW_MIN( lS,(len)-1 )]=0;}}
// #else

// #define PW_STRCPY( strD, len, strS ) {		\
// 	if( len > 0 ){							\
// 	GInt32 lS = strlen(strS);				\
// 	lS = PW_MIN( lS, (len)-1 );				\
// 	strncpy( (strD), (strS), (len)-1 );		\
// 	(strD)[lS]=0;}}
// // #endif

// #define PW_STRCAT( strD, len, strS ) {					\
// 	if( len > 0 ){										\
// 	GInt32 lS = strlen(strS), lD = strlen(strD);		\
// 	strncat( (strD), (strS), PW_MAX( (len)-1-lD, 0 ) );	\
// 	(strD)[PW_MIN(lS+lD,(len)-1)]=0; } }

// #ifdef _WIN32
// #define PW_STRCAT( strD, len, strS ) {strcat_s( strD, len, strS );}
// #else
// #define PW_STRCAT( strD, len, strS ) ( strncat( strD, strS, len ) )
// #endif
// 
// #ifdef _WIN32
// #define PW_STRCPY( strD, len, strS ) {strcpy_s( strD, len, strS );}
// #else
// // #define PW_STRCPY( strD, len, strS ) {	\
// // 	int _t = 0;							\
// // 	char *_pS = strS, *_pD = strD;		\
// // 	while( _t < len-1 && *_pS ){		\
// // 		*_pD = *_pS;	_pS++; _pD++;	\
// // 	}									\
// // 	*_pD = 0;							\
// // }
// #define PW_STRCPY( strD, len, strS ) ( strncpy( strD, strS, len ) )
// #endif

#ifdef _WIN32
#define PW_STRTOK( szSrc, szKey, szLft, szRgt ) {szLft = strtok_s( szSrc, szKey, &szRgt );}
#else
#define PW_STRTOK( szSrc, szKey, szLft, szRgt ) {szLft = strtok_r( szSrc, szKey, &szRgt );}
#endif

#define PW_DELETE( pBuff ){			\
	if( pBuff ) delete( pBuff );	\
	pBuff = 0;						\
}
// #define PW_DELETE( pBuff ){			\
// 	GVoid *pT = pBuff;				\
// 	pBuff = 0;						\
// 	if( pT ) delete( pT );			\
// }

#define PW_NEW_AST( pstBuff, STRUCT_TYPE, lNum ){	\
	if( !pstBuff ) pstBuff = new STRUCT_TYPE[lNum];	\
}

#define PW_DELETE_AST( pstBuff ){		\
	if( pstBuff ) delete[]( pstBuff );	\
	pstBuff = 0;						\
}
// #define PW_DELETE_AST( pstBuff ){		\
// 	GVoid*pT = pstBuff;						\
// 	pstBuff = 0;							\
// 	if( pT ) delete[]( pT );				\
// }

#define PW_STR_2_UI32( cChar0, cChar1, cChar2, cChar3, ulValue ){	\
	ulValue = (cChar3);												\
	ulValue = ( ulValue << 8 ) + (cChar2);							\
	ulValue = ( ulValue << 8 ) + (cChar1);							\
	ulValue = ( ulValue << 8 ) + (cChar0);							\
}
#define PW_UI32_2_STR( cChar0, cChar1, cChar2, cChar3, ulValue ){	\
	cChar0 = ulValue & 0xff;										\
	cChar1 = ( ulValue <<16 ) >> 8;									\
	cChar2 = ( ulValue <<8 ) >> 16;									\
	cChar3 = ulValue >> 24;											\
}
#define PW_STR_2_UI64( cChar0, cChar1, cChar2, cChar3, cChar4, cChar5, cChar6, cChar7, ullValue ){	\
	ullValue = (cChar7);											\
	ullValue = ( ullValue << 8 ) + (cChar6);						\
	ullValue = ( ullValue << 8 ) + (cChar5);						\
	ullValue = ( ullValue << 8 ) + (cChar4);						\
	ullValue = ( ullValue << 8 ) + (cChar3);						\
	ullValue = ( ullValue << 8 ) + (cChar2);						\
	ullValue = ( ullValue << 8 ) + (cChar1);						\
	ullValue = ( ullValue << 8 ) + (cChar0);						\
}
#define PW_UI64_2_STR( cChar0, cChar1, cChar2, cChar3, cChar4, cChar5, cChar6, cChar7, ullValue ){	\
	cChar0 = ulValue & 0xff;										\
	cChar1 = ( ulValue <<48 ) >> 8;									\
	cChar2 = ( ulValue <<40 ) >> 16;								\
	cChar3 = ( ulValue <<32 ) >> 24;								\
	cChar4 = ( ulValue <<24 ) >> 32;								\
	cChar5 = ( ulValue <<16 ) >> 40;								\
	cChar6 = ( ulValue <<8 ) >> 48;									\
	cChar7 = ulValue >> 56;											\
}
#endif
