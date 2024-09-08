#include "pw_base_api.h"
#include <string.h>
#include <stdlib.h>

GVM_SCREEN *pwbCreateImage( GInt32 lWidth, GInt32 lHeight, GInt32 lColorSpace )
{
    GVM_SCREEN *pstImage = GNull;
    GUInt32 ulBuffSize=0, ulPitch0=0, ulPitch1=0;
    GUInt8 *pbyBuff = GNull;
    ulBuffSize += sizeof(GVM_SCREEN);
    switch( lColorSpace )
    {
    case VM_FORMAT_YUV420_P:
        ulPitch0 = ( lWidth + 3 ) & 0xfffffffc;
        ulPitch1 = ( (lWidth >> 1) + 3 ) & 0xfffffffc;
        ulBuffSize += ulPitch0 * lHeight;
        ulBuffSize += ulPitch1 * lHeight;
        break;
    case VM_FORMAT_NV12:
        ulPitch0 = ( lWidth + 3 ) & 0xfffffffc;
        ulPitch1 = ( lWidth + 3 ) & 0xfffffffc;
        ulBuffSize += ulPitch0 * lHeight;
        ulBuffSize += ulPitch1 * lHeight >> 1;
        break;
    case VM_FORMAT_R8G8B8A8:
        ulPitch0 = lWidth * 4;
        ulBuffSize += ulPitch0 * lHeight;
        break;
    case VM_FORMAT_B8G8R8A8:
        ulPitch0 = lWidth * 4;
        ulBuffSize += ulPitch0 * lHeight;
        break;
    case VM_FORMAT_R8G8B8:
        ulPitch0 = ( lWidth * 3 + 3) & 0xfffffffc;
        ulBuffSize += ulPitch0 * lHeight;
        break;
    case VM_FORMAT_B8G8R8:
        ulPitch0 = ( lWidth * 3 + 3) & 0xfffffffc;
        ulBuffSize += ulPitch0 * lHeight;
        break;
    case VM_FORMAT_GRAY8:
        ulPitch0 = ( lWidth + 3) & 0xfffffffc;
        ulBuffSize += ulPitch0 * lHeight;
        break;
    }

    UNGO( pbyBuff, ERREXT, (GUInt8*)malloc(ulBuffSize), GNull );

//     UNGO( pbyBuff, ERREXT, new GUInt8[ulBuffSize], GNull );

    pstImage = (GVM_SCREEN*)pbyBuff;
    pstImage->pBuffHandle = pbyBuff;
    pbyBuff += sizeof(GVM_SCREEN);
    pstImage->lWidth = lWidth;
    pstImage->lHeight = lHeight;
    pstImage->ImgFormat = lColorSpace;

    switch( lColorSpace )
    {
    case VM_FORMAT_YUV420_P:
        pstImage->PixelArray.planar.lLineBytesArray[0] = ulPitch0;
        pstImage->PixelArray.planar.lLineBytesArray[1] = ulPitch1;
        pstImage->PixelArray.planar.lLineBytesArray[2] = ulPitch1;
        pstImage->PixelArray.planar.pPixelArray[0] = pbyBuff;
        pbyBuff += ulPitch0 * lHeight;
        pstImage->PixelArray.planar.pPixelArray[1] = pbyBuff;
        pbyBuff += ulPitch1 * lHeight >> 1;
        pstImage->PixelArray.planar.pPixelArray[2] = pbyBuff;
        pbyBuff += ulPitch1 * lHeight >> 1;
        memset( (GUInt8*)pstImage->PixelArray.planar.pPixelArray[0], 0,
            pstImage->PixelArray.planar.lLineBytesArray[0] * pstImage->lHeight );
        memset( (GUInt8*)pstImage->PixelArray.planar.pPixelArray[1], 128,
            pstImage->PixelArray.planar.lLineBytesArray[1] * pstImage->lHeight / 2 );
        memset( (GUInt8*)pstImage->PixelArray.planar.pPixelArray[2], 128,
            pstImage->PixelArray.planar.lLineBytesArray[2] * pstImage->lHeight / 2 );
        break;
    case VM_FORMAT_NV12:
        pstImage->PixelArray.planar.lLineBytesArray[0] = ulPitch0;
        pstImage->PixelArray.planar.lLineBytesArray[1] = ulPitch1;
        pstImage->PixelArray.planar.pPixelArray[0] = pbyBuff;
        pbyBuff += ulPitch0 * lHeight;
        pstImage->PixelArray.planar.pPixelArray[1] = pbyBuff;
        memset( (GUInt8*)pstImage->PixelArray.planar.pPixelArray[0], 0,
            pstImage->PixelArray.planar.lLineBytesArray[0] * pstImage->lHeight );
        memset( (GUInt8*)pstImage->PixelArray.planar.pPixelArray[1], 128,
            pstImage->PixelArray.planar.lLineBytesArray[1] * pstImage->lHeight / 2 );
        break;
    case VM_FORMAT_R8G8B8A8:
    case VM_FORMAT_B8G8R8A8:
    case VM_FORMAT_R8G8B8:
    case VM_FORMAT_B8G8R8:
    case VM_FORMAT_GRAY8:
        pstImage->PixelArray.chunky.lLineBytes = ulPitch0;
        pstImage->PixelArray.chunky.pPixel = pbyBuff;
        memset( (GUInt8*)pstImage->PixelArray.chunky.pPixel, 0,
            pstImage->PixelArray.chunky.lLineBytes * pstImage->lHeight );
        break;
    }

ERREXT:
    return pstImage;
}

GVoid pwbReleaseImage( GVM_SCREEN *pstImage )
{
    if( pstImage && pstImage->pBuffHandle )
//         delete( (GUInt8*)pstImage->pBuffHandle );
        free( (GUInt8*)pstImage->pBuffHandle );
}
