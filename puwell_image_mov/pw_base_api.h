

#ifndef PW_BASE_API_H
#define PW_BASE_API_H

#ifdef __ANDROID__
    #include "inc/pw_datatype.h"
    #include "inc/pw_typedef.h"
    #include "inc/pwerror.h"
    #include "inc/pwmacro.h"
#else
    #include "pw_datatype.h"
    #include "pw_typedef.h"
    #include "pwerror.h"
    #include "pwmacro.h"
#endif

#ifdef _WIN32
#define PWBPRINTF sprintf_s
#else
#define PWBPRINTF snprintf

#endif
/*-- Time --*/
GVoid    pwbGetGrnWshTime( PW_TIME &stTime );
GVoid    pwbGetGrnWshTime( PW_TIME2 &stTime );
GInt64    pwbmkgmtime( struct tm *unixdate );
GInt32    pwbGetLocalTimeOff();
GUInt32 pwbDate_2_DayOff2010( PW_TIME2 *pstDate );
GVoid    pwbDayOff2010_2_Date( GUInt32 ulDayOff, PW_TIME2 *pstDate );
GVoid    pwbGetLocalTime( PW_TIME2 *pstTime );
GUInt64 pwbGetMsec64();        //    从1970年起
GVoid    pwbCompleteTime( PW_TIME2 *pstTime );
GBool    TimeGw2Dev( GInt32 lTimeOff, GUInt64 ullTimeGw, PW_TIME2 *pstTime );
GBool    TimeDev2Gw( GInt32 lTimeOff, GUInt64 *pullTimeGw, PW_TIME2 *pstTime );
GVoid    pwbTime2GoTime( PW_TIME2 *pstTime2, PW_TIME *pstTime );
GVoid    pwbTimeGoTime2( PW_TIME2 *pstTime2, PW_TIME *pstTime );
GBool    pwbCheckTimeRange( PW_TIME_RG *pstTimeRange, GBool bAutoCorect = GTrue );
GVoid    pwbTimeRangeGoTime( PW_TIME_RG *pstTimeR, PW_TIME *pstTimeS, PW_TIME *pstTimeE );
GVoid    pwbTimeRangeGoTime2( PW_TIME_RG *pstTimeR, PW_TIME2 *pstTimeS, PW_TIME *pstTimeE );
/*-- String --*/
GBool    pwbSafeStrCpy( GChar *szDst, GInt32 lDstLen, GChar *szSrc );
GInt32    pwb_strlen ( GChar* szSrc );
GVoid    pwbStrlwr( GChar *szStr );
GVoid    pwbStrupr( GChar *szStr );
GInt32    pwbUtf8_len( GChar *szUTF8 );
GBool    pwbUtf8_2_Unicode( GChar *szUTF8, GUInt16 *szwUnicode, GInt32 lUnicodeLen );
GBool    pwbCheckTailSlashe( GChar *szPath, GUInt32 ulLen );
/*--
    * 1字节 0xxxxxxx
    * 2字节 110xxxxx 10xxxxxx
    * 3字节 1110xxxx 10xxxxxx 10xxxxxx
    * 4字节 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    * 5字节 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    * 6字节 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    return    0: OK; -1: 运行错误; -2: 有超位数字符
    e.g.  数据库只支持3字节以内的字符，lMaxBits = 3, bOnlyCheckBits = GTrue
--*/
GInt32    pwbCheckBitsUTF8( GChar *szStr, GUInt32 lMaxBits, GBool bOnlyCheckBits );


GChar*    pwbUrlEncode( GChar *szSrc );

/*-- File --*/
GBool    pwbMakeSureFolder( GChar *szDirectoryPath );
GBool    pwbChangeExtension( GChar *szFilePath, GInt32 lBufflen, GChar *szNewExten );
GBool    pwbDeleteFolder( GChar *szFolderPath, GBool bOnlyData );
GBool    pwbDeleteFile( GChar *szFilePath );

/*-- Image --*/
enum ROTATE_IMG_CFG
{
    ROTATE_IMG_CFG_90        =0,
    ROTATE_IMG_CFG_270        ,
    ROTATE_IMG_CFG_90_MIR    ,
    ROTATE_IMG_CFG_270_MIR    ,
    ROTATE_IMG_CFG_180         =10,
    ROTATE_IMG_CFG_180_MIR    ,
    ROTATE_IMG_CFG_MIR        ,
};
GVM_SCREEN* pwbCreateImage( GInt32 lWidth, GInt32 lHeight, GInt32 lColorSpace );
GVM_SCREEN* pwbMakeSureImage( GVM_SCREEN *pstImage, GInt32 lWidth, GInt32 lHeight, GInt32 lImgFormat );
GVoid    pwbReleaseImage( GVM_SCREEN *pstImage );
GBool    pwbZeroImage( GVM_SCREEN *pstImage );
GBool    pwbGetImageRate( GSIZE stImageSize, GSIZE *pstRate );
GInt32    pwbLoadBMP( char *szFile, GInt8 **pBuffer, GInt32 *nWidth, GInt32 *nHeight, GInt32 *nBitCount );
GInt32    pwbSaveBMP( char* szFile, GInt8 *pData, GInt32 nWidth, GInt32 nHeight, GInt32 nBitCount );

GBool    pwbCopyRoi( GVM_SCREEN* pstSrc, GVM_SCREEN** pstDst, GRECT *pstRoi );

GBool    pwbResizeYUV420P( GVM_SCREEN *pstSrc, GVM_SCREEN *pstDst );

GBool    pwbRotateImage_YUV420P( GVM_SCREEN *pstSrcImage, GVM_SCREEN *pstDstImage, ROTATE_IMG_CFG eCfg );

GBool    pwbYUV420P_2_BGRA_S( GVM_SCREEN* pstYUV420P, GVM_SCREEN* pstBGRA, GInt32 lScale );
GBool    pwbNV12_2_BGRA_S( GVM_SCREEN* pstNV12, GVM_SCREEN* pstBGRA, GInt32 lScale );
GBool    pwbNV12_2_YUV420P( GVM_SCREEN* pstNV12, GVM_SCREEN* pstYUV420P );
GBool    pwbYUV420P_2_NV12( GVM_SCREEN* pstYUV420P, GVM_SCREEN* pstNV12 );
/*-- pstDstImg->lWidth = pstSrcImg->lWidth >> ulScaleStep[0,2] --*/
GBool    pwbYUV420P_Copy_S( GVM_SCREEN* pstSrcImg, GVM_SCREEN* pstDstImg, GUInt32 ulScaleStep );
GBool    pwbNV12_Copy_S( GVM_SCREEN* pstSrcImg, GVM_SCREEN* pstDstImg, GUInt32 ulScaleStep );
GBool    pwbGray_Copy_S( GVM_SCREEN* pstSrcImg, GVM_SCREEN* pstDstImg, GUInt32 ulScaleStep );
GBool    pwbYUV420P_2_NV12_2S( GVM_SCREEN* pstYUV420P, GVM_SCREEN* pstNV12, GInt32 lScale );

GBool    pwbLineYUV420P( GVM_SCREEN *pstYUV420P, GLINE *pstLine, GPOINT3D *pstYUV, GInt32 lThick );
GVoid    pwbVertexAngle( GFPOINT3D *pstVertex, GFloat *pfPan_A, GFloat *pfTilt_A );

/*-- Encode & Decode --*/

#define  PWB_CODE_STEP    128
#define  PWB_CODE_MOD    0x7f

class PWB_RC4
{
public:
    PWB_RC4();
    ~PWB_RC4();
    GVoid rc4Init( GUInt8 *pbyKey, GInt32 lLen );
    GVoid rc4Crypt( GUInt8* pbyData, GInt32 lLen );
private:
    GUInt8 m_pbyPro[PWB_CODE_STEP];
    GUInt8 m_pbyTmp[PWB_CODE_STEP];
};

typedef struct _pw_mask_zip_
{
    GUInt64    ullVer        :2;        //    = 0
    GUInt64    ullOriWid    :13;
    GUInt64    ullOriHei    :13;
    GUInt64    ullCodeLen    :25;
    GUInt64    ullBVerScan    :1;
    GUInt64    ullExtern    :10;
}PW_MASK_ZIP;

typedef struct _pw_mask_zip_s_
{
    GUInt32    ullVer        :2;        //    = 1
    GUInt32    ullOriWid    :7;
    GUInt32    ullOriHei    :7;
    GUInt32    ullCodeLen    :15;
    GUInt32    ullBVerScan    :1;
}PW_MASK_ZIP_S;

/*--
* if( code & 128 )        mask = code - 128;    bit[7] 为1， codebit[0~6]表示原mask的值
* else if( code == 0 )    line = {0}            code为0， 该行全为0
* else                    line += code * {0}    bit[7] 为0， 该行当前位置补codebit[0~6]个0
--*/
/*-- Decompress a code stream back to a mask image --*/
GBool pwbMaskZip_2Mask( GUInt8 *pbyCodeBuff, GInt32 lCodeBufLen, GVM_SCREEN **ppstMask, PW_MASK_ZIP *pstInfo = GNull );
/*--
    Compress a mask(8bits gray image) to a code stream, the value in mask must < 128
    lCodeBufLen >= pstMask->lWidth * pstMask->lHeight
--*/
GBool pwbMaskZip_2Code( GVM_SCREEN *pstMask, GUInt8 *pbyCodeBuff, GInt32 lCodeBufLen,
                        GBool bVerScan, PW_MASK_ZIP *pstInfo = GNull );

/*-- G711 --*/

GInt32    xmn_G711a_Decode( GUInt8 *pbySrc, GShort *psPCM, GInt32 lSrcLen, GUInt32 ulIgnoreLv );
GInt32    xmn_G711a_DecodeST( GUInt8 *pbySrc, GFloat *pfPCM, GInt32 lSrcLen, GUInt32 ulIgnoreLv );
GInt32    xmn_G711u_Decode( GUInt8 *pbySrc, GShort *psPCM, GInt32 lSrcLen, GUInt32 ulIgnoreLv );







#endif
