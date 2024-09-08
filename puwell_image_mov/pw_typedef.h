
/*----------------------------------------------------------------------------------------------
* 2014 07 02 lgy	Add VM_PANO_2560_1920	VM_PANO_4000_3000
* 2014 11 04 lgy	Check all mirror
*---------------------------------------------------------------------------------------------*/

#ifndef __PUWELLTYPEDEF_H__
#define __PUWELLTYPEDEF_H__

#include "pw_datatype.h"
//////////////////////////////////////////////////////////////////////////
/*			Common Define												*/

typedef struct __tag_rect
{
	GInt32 left;
	GInt32 top;
	GInt32 right;
	GInt32 bottom;
} GRECT, *LPGRECT;

typedef struct __view_port
{
	float x;
	float y;
	float width;
	float height;
} VIEWPORT;

typedef struct __tag_frect
{
	GFloat left;
	GFloat top;
	GFloat right;
	GFloat bottom;
} GFRECT, *LPGFRECT;

typedef struct __tag_srect
{
	GInt16 left;
	GInt16 top;
	GInt16 right;
	GInt16 bottom;
} GSRECT, *LPGSRECT;

typedef struct __tag_point
{ 
	GInt32 x; 
	GInt32 y; 
} GPOINT, *LPGPOINT;

typedef struct __tag_spoint
{ 
	GInt16 x; 
	GInt16 y; 
} GSPOINT, *LPGSPOINT;

typedef struct __tag_cpoint
{
	GUInt8 x;
	GUInt8 y;
} GCPOINT, *LPGCPOINT;

typedef struct __tag_fpoint
{ 
	GFloat x; 
	GFloat y; 
} GFPOINT, *LPGFPOINT;

typedef struct __tag_size
{ 
	GInt32 Wid; 
	GInt32 Hei; 
} GSIZE, *LPGSIZE;

typedef struct __tag_fsize
{ 
	GFloat Wid; 
	GFloat Hei; 
} GFSIZE, *LPGFSIZE;

typedef struct __tag_point3D
{ 
	GInt32 x;								
	GInt32 y;							
	GInt32 z;							
} GPOINT3D, *LPGPOINT3D;

typedef struct __tag_color
{ 
	GUInt8 b;								
	GUInt8 g;							
	GUInt8 r;	
	GUInt8 a;
} GCOLOR, *LPGCOLOR;

typedef struct __tag_fpoint3D
{ 
	GFloat x;
	GFloat y;
	GFloat z;
} GFPOINT3D, *LPFGPOINT3D;

typedef struct __tag_fpoint4D
{ 
	GFloat x;
	GFloat y;
	GFloat z;
	GFloat a;
} GFPOINT4D, *LPFGPOINT4D;

typedef struct __tag_line
{ 
	GPOINT P1; 
	GPOINT P2; 
} GLINE, *LPGLINE;

typedef struct __tag_sline
{
	GSPOINT P1;
	GSPOINT P2;
} GSLINE, *LPGSLINE;

typedef struct __tag_fline
{ 
	GFPOINT P1; 
	GFPOINT P2; 
} GFLINE, *LPGFLINE;

typedef struct __tag_fline3d
{ 
	GFPOINT3D P1; 
	GFPOINT3D P2; 
} GFLINE3D, *LPGFLINE3D;

typedef struct __tag_elli
{
	GDouble Para[6];	//	Para[0]*X^2 + Para[1]*X + Para[2]*Y^2 + Para[3]*Y + Para[4]*X*Y = Para[5]
	GFPOINT Center;
	GFLINE LongAxis;
	GFLINE ShortAxis;
} GCOMELLI;

typedef struct _pwfspinsex
{
	GFPOINT		ScreenPoint; 
	GSIZE		ScreenSize;
}GPOINTINSCREEN;

typedef struct _pwlspinsex
{
	GPOINT		ScreenPoint; 
	GSIZE		ScreenSize;
}GLPOINTINSCREEN;

typedef enum pw_str_type_
{
	charpoint = 0,
	bytepoint = 1,
	wcharpoint = 2,
} GSTRTYPE;

typedef struct _pwstr
{
	GInt32		StrLen; 
	GSTRTYPE	StrType;
	union
	{
		char	cpStr[64];
		unsigned char	bpStr[64];
		unsigned short	wpStr[32];
	}StringP;
}GString;

#define FS_MAX_CORRECT_GROUP_POINT_NUM 128
#define FS_MAX_SERIAL_LIST		4
#define FS_MAX_SERIAL_STRLEN	256
#define FS_MAX_MSGRETURN		256

typedef struct _fs_single_serialcom
{
	GInt32 StrLenth;
	GUInt8 SerialComStr[FS_MAX_SERIAL_STRLEN];
	GUInt8 MsgReturn[FS_MAX_MSGRETURN];
	GInt32 WaitMilliSecond;
}GFSSINGLESERIALCOM;

typedef struct _fs_serialcomlist
{
	GInt32 SingleComNum;
	GFSSINGLESERIALCOM SerialCom[FS_MAX_SERIAL_LIST];
}GFSSERIALCOMLIST;

//////////////////////////////////////////////////////////////////////////
/*			Image / Frame Define										*/
#ifndef VM_FORMAT_
#define VM_FORMAT_
#define VM_FORMAT_PLANAR		0x0100
#define VM_FORMAT_BEPLANAR_		0x0f00

#define VM_FORMAT_GRAY8			0x001		//	Y
#define VM_FORMAT_B8G8R8		0x002		//	B8	G8	R8
#define VM_FORMAT_B8G8R8A8		0x003		//	B8	G8	R8	A8
#define VM_FORMAT_R8G8B8		0x004		//	R8	G8	B8
#define VM_FORMAT_R8G8B8A8		0x005		//	R8	G8	B8	A8
#define VM_FORMAT_B5G6R5		0x006		//	B5	G6	R5
#define VM_FORMAT_R5G6B5		0x007		//	R5	G6	B5
#define VM_FORMAT_YUY2			0x011		//	Y0	U0	Y1	V0	Y2	U1	Y3	V1
#define VM_FORMAT_UYVY			0x012		//	Y0	U0	Y1	V0	Y2	U1	Y3	V1
#define VM_FORMAT_YUV422		0x013		//	Y0	U0	Y1	V0

#define VM_FORMAT_YUV420_P		(0x011 | VM_FORMAT_PLANAR)		
#define VM_FORMAT_YUV12420_P	(0x012 | VM_FORMAT_PLANAR)		
#define VM_FORMAT_YUV422_P		(0x013 | VM_FORMAT_PLANAR)		
#define VM_FORMAT_NV12          (0x014 | VM_FORMAT_PLANAR)
#endif

typedef struct _vm_screen
{
	GInt32		lWidth;
	GInt32		lHeight;
	GInt32		ImgFormat;
	void		*pBuffHandle;
	union
	{
		struct  
		{
			GInt32	lLineBytes;
			void	*pPixel;	
		}chunky;
		struct  
		{
			GInt32	lLineBytesArray[4];
			void	*pPixelArray[4];
		}planar;
	}PixelArray;
}GVM_SCREEN, *LPGVM_SCREEN;

typedef union _pw_data_un_
{
	GUInt64		Data_UI64;
	GInt64		Data_I64;
	GUInt32		Data_UI32[2];
	GInt32		Data_I32[2];
	GUInt16		Data_UI16[4];
	GInt16		Data_I16[4];
	GUInt8		Data_UI8[8];
	GInt8		Data_I8[8];
	GDouble		Data_D;
	GFloat		Data_F[2];	
}PW_DATA_UN;

typedef struct _pw_time_ 
{
	GUInt32 ulSecond	:6;		//	0-59 
	GUInt32 ulMinute	:6;		//	0-59 
	GUInt32 ulHour		:5;		//	0-23 
	GUInt32 ulDay		:5; 	//	1-31 
	GUInt32 ulMonth		:4; 	//	1-12 
	GUInt32 ulYear		:6; 	//	2000-2063 	
} PW_TIME,*LP_PW_TIME;

typedef struct _pw_time_2_
{
	GUInt32 ulYear;		// AD
	GUInt32 ulMonth;	// 1~12
	GUInt32 ulDay;		// 1~31
	GUInt32 ulHour;		// 0~23
	GUInt32 ulMinute;	// 0~59
	GUInt32 ulSecond;	// 0~59
	GUInt32 ulDayW;		// 0~6
	GUInt32 ulDayY;		// 0~365
}PW_TIME2; 

typedef struct _pw_time_clock_
{
	GUInt32 ulMsec		:10;	//	0-999 
	GUInt32 ulSecond	:6;		//	0-59 
	GUInt32 ulMinute	:6;		//	0-59 
	GUInt32 ulHour		:5;		//	0-23 
	GUInt32 ulExtern	:5;	
}PW_TIMEC; 

typedef struct _pw_time_rg_
{
	//Date; 隶属日期
	GUInt64 ullYear		:21; 	//	AD
	GUInt64 ullMonth	:4; 	//	1-12
	GUInt64 ullDay		:5; 	//	1-31
	//Start time; 起始时间
	GUInt64 ullHour_S	:5;		//	0-23
	GUInt64 ullMin_S	:6;		//	0-59
	GUInt64 ullSec_S	:6;		//	0-59
	//End time; 终止时间
	GUInt64 ullHour_E	:5;		//	0-23
	GUInt64 ullMin_E	:6;		//	0-59
	GUInt64 ullSec_E	:6;		//	0-59

}PW_TIME_RG;

#define VM_LENS_
#define VM_LENS_SHOWMO_00		0
#define VM_LENS_K103			1
#define VM_LENS_K104			2
#define VM_LENS_HH701			3	//	红海701 2.8mm
#define VM_LENS_FREE			4
#define VM_LENS_COUNT			5
#define VM_LENS_MAX_COUNT		20
#define VM_LENS_NORMAL			31

#define HD_LENS_00_65			0
#define HD_LENS_N10_65			1
#define HD_LENS_N20_70			2
#define HD_LENS_00_75			3
#define HD_LENS_N20_45			4
#define HD_LENS_MASK			(2<<6)	
#define HD_LENS_COUNT			5

//////////////////////////////////////////////////////////////////////////
//	2015 12 09 lgy add new definition

typedef struct _pw_pano_type_00_
{
	GUInt32		ulDefNo		:4;		//	0x01
	GUInt32		ulResRateW	:5;		//	Resolution is lResRateW : ullResRateH; Follow the frame size if either of them is 0
	GUInt32		ulResRateH	:5;
	GUInt32		ulWidth_8	:11;	//	Width/8
	GUInt32		ulLensNo	:6;		//	VM_LENS_
	GUInt32		ulExtern	:1;	
}GVM_PANO_TYPE_00;

#define PW_LENS_TYPE_BC_PTZ		0
#define PW_LENS_TYPE_FISH		1
#define PW_LENS_TYPE_HD			2
#define PW_LENS_TYPE_WA			3	//	Wide angle
#define PW_LENS_TYPE_PW4PI		4	//	4πUnit
#define PW_LENS_TYPE_BC_VIR		5	//	vir_project
#define PW_LENS_TYPE_BC_NUL		6	//	Noraml box cam 
#define PW_LENS_TYPE_MUL_LENS	7	//	Multiple lens zoom


typedef struct _pw_pano_type_01_
{
	GUInt64		ullDefNo			:4;		//	0x02
	GUInt64		ullResRateW			:5;		//	Resolution is lResRateW : ullResRateH; Follow the frame size if either of them is 0
	GUInt64		ullResRateH			:5;
	GUInt64		ullWidth_8			:11;	//	Width/8
	GUInt64		ullRadius_4			:11;	//	Radius/4
	GUInt64		ullLensType			:3;		//	PW_LENS_TYPE_FISH || PW_LENS_TYPE_HD
	GUInt64		ullLensNo			:6;		//	VM_LENS_
	GUInt64		ullBeCenOffX_Minus	:1;		//	sign( ( CenterX - Width / 2 ) / 2 )
	GUInt64		ullCenterOffX_2		:5;		//	fabs( ( CenterX - Width / 2 ) / 2 )
	GUInt64		ullBeCenOffY_Minus	:1;		//	sign( ( CenterY - Height / 2 ) / 2 )
	GUInt64		ullCenterOffY_2		:5;		//	fabs( ( CenterY - Height / 2 ) / 2 )
	GUInt64		ullBeVerSensor		:1;		//	
	GUInt64		ullExten			:6;		//	
}GVM_PANO_TYPE_01;

typedef struct _pw_pano_type_ptv_	//	For lens correct BC
{
	GUInt64		ullDefNo			:4;		//	0x03
	GUInt64		ullLensType			:3;		//	PW_LENS_TYPE_BC_VIR
	GUInt64		ullLensNo			:16;	//	Lens_correct_pro_ID
	GUInt64		ullResRateW			:5;		//	Resolution is lResRateW : ullResRateH; Follow the frame size if either of them is 0
	GUInt64		ullResRateH			:5;
	GUInt64		ullWidth_8			:11;	//	Width/8
	GUInt64		ullViewRange_X10	:11;	//	0~180 * 10
	GUInt64		ullBeVerSensor		:1;		//	
	GUInt64		ullExten			:8;		//	
}GVM_PANO_TYPE_PTV;


typedef struct _pw_pano_type_pt_
{
	GUInt64		ullDefNo			:4;		//	0x04
	GUInt64		ullLensType			:3;		//	PW_LENS_TYPE_BC_PTZ
	GUInt64		ullLensNo			:6;		//	VM_LENS_NORMAL
	GUInt64		ullResRateW			:5;		//	Resolution is lResRateW : ullResRateH; Follow the frame size if either of them is 0
	GUInt64		ullResRateH			:5;
	GUInt64		ullWidth_8			:11;	//	Width/8
	GUInt64		ullViewRange_X10	:11;	//	0~180 * 10
	GUInt64		ullViewRange_Y10	:11;	//	0~180 * 10
	GUInt64		ullBeVerSensor		:1;		//	
	GUInt64		ullExten			:7;		//	
}GVM_PANO_TYPE_PT;



typedef struct _pw_pano_type_rollens_
{
	GUInt64		ullDefNo			:4;		//	0x05
	GUInt64		ullLensType			:3;		//	PW_LENS_TYPE_BC_VIR
	GUInt64		ullLensNo			:7;		//	Lens_correct_pro_ID
	GUInt64		ullDefWidth_8		:11;	//	Width/8
	GUInt64		ullDefHeight_8		:10;	//	Height/8
	GUInt64		ullViewRange_X10	:11;	//	0~180 * 10
	GUInt64		ullExten			:18;	//	
}GVM_PANO_TYPE_ROLLENS;

// typedef struct _pw_pano_type_mlz_
// {
// 	GUInt64		ullDefNo			:4;		//	0x06
// 	GUInt64		ullLensType			:3;		//	PW_LENS_TYPE_MUL_LENS
// 	GUInt64		ullResRateW			:5;		//	Resolution is lResRateW : ullResRateH; Follow the frame size if either of them is 0
// 	GUInt64		ullResRateH			:5;
// 	GUInt64		ullWidth_8			:11;	//	Width/8
// 	GUInt64		ullViewR1_X4		:9;		//	0~127 * 4
// 	GUInt64		ullViewR2_X4		:9;		//	0~127 * 4
// 	GUInt64		ullViewR3_X4		:8;		//	0~63 * 4
// 	GUInt64		ullViewR4_X4		:7;		//	0~31 * 4
// 	GUInt64		ullExten			:3;	//	
// }GVM_PANO_TYPE_MLZ;

// typedef struct _pw_pano_type_ptfish_
// {
// 	GUInt64		ullDefNo			:4;		//	0x07
// 	GUInt64		ullLensType			:3;		//	PW_LENS_TYPE_FISH
// 	GUInt64		ullLensNo			:6;		//	VM_LENS_
// 	GUInt64		ullDefResRateW		:5;		//	Resolution is lResRateW : ullResRateH; Follow the frame size if either of them is 0
// 	GUInt64		ullDefResRateH		:5;
// 	GUInt64		ullDefWidth_8		:11;	//	Width/8
// 	GUInt64		ullVirResRateW		:5;		//	Resolution is lResRateW : ullResRateH; Follow the frame size if either of them is 0
// 	GUInt64		ullVirResRateH		:5;
// 	GUInt64		ullVirWidth_8		:9;		//	Width/8
// 	GUInt64		ullFillWid_or_Hei	:1;		//	0 : Radius = DefWidth / 2;	1 : Radius = DefHeight / 2; 
// 	GUInt64		ullBeVerSensor		:1;		//	
// 	GUInt64		ullExten			:9;		//	
// }GVM_PANO_TYPE_PTFISH;

typedef struct _pw_pt_range_svr_
{
	GUInt32		ulPanRange		:9;		//	0~360
	GUInt32		ulMinTilt		:9;		//	-180~180 + 180
	GUInt32		ulMaxTilt		:9;		//	-180~180 + 180
	GUInt32		ulExtern		:5;		//	-180~180 + 180
}GVM_PT_RANGE_SVR;

typedef struct _pw_ptz_range_ipc_
{
	/*-- Range of pan is symmetrical to 180 such as (5~355) --*/
	GUInt64		ullMinPan_10		:11;	//	0~1800 = 1800 - ( GVM_PTZ_RANGE_SVR.ulMaxPan >> 1 )
	GUInt64		ullMaxPan_10		:11;	//	0~1800 = ( GVM_PTZ_RANGE_SVR.ulMaxPan >> 1 )
	GUInt64		ullMinTilt_10		:12;	//	( -180~180 + 180 ) * 10 = ( GVM_PTZ_RANGE_SVR.ulMinTilt ) * 10
	GUInt64		ullMaxTilt_10		:12;	//	( -180~180 + 180 ) * 10 = ( GVM_PTZ_RANGE_SVR.ulMaxTilt ) * 10
	GUInt64		ullMaxZoom_10		:7;		//	Reserved
	GUInt64		ullExten			:11;	
}GVM_PTZ_RANGE_IPC;

GInt32 pwGetPanoTypeVer( GUInt64 dwPanoramicType );
GBool pwGetPanoLens( GUInt64 dwPanoramicType, GInt32 *plLensType, GInt32 *plLensNo );
GBool pwGetPanoSize( GUInt64 dwPanoramicType, GSIZE *pstFrameSize );
GBool pwGetImageRate( GSIZE stImageSize, GSIZE *pstRate );

//////////////////////////////////////////////////////////////////////////
/*			Panoramic Type Define										*/
#define VM_PANO_RESO_MASK		0xff000	
#define VM_PANO_768_576			0x01000	
#define VM_PANO_1280_1024		0x02000	
#define VM_PANO_2048_1536		0x03000	
#define VM_PANO_1280_960		0x04000	
#define VM_PANO_2448_2048		0x05000	
#define VM_PANO_1920_1080		0x06000	
#define VM_PANO_1024_1024		0x07000	
#define VM_PANO_1536_1536		0x08000
#define VM_PANO_2560_1920		0x09000
#define VM_PANO_4000_3000		0x0a000
#define VM_PANO_1280_720		0x10000

#define VM_PANO_MIRROR_MASK		0x0ff0	
#define VM_PANO_MIRROR_STEP		0x04
#define VM_PANO_00_65			0x0000	
#define VM_PANO_N10_65			0x0010	
#define VM_PANO_N20_70			0x0020	
#define VM_PANO_00_75			0x0030	
#define VM_PANO_N20_45			0x0040	
#define VM_PANO_N02_90F			0x0200	
#define VM_PANO_N00_120W		0x0300	
#define VM_PANO_N00_90W			0x0400	

#define VM_BITERATE_1			300
#define VM_BITERATE_2			600
#define VM_BITERATE_3			1200
#define VM_BITERATE_4			2400
#define VM_BITERATE_5			4800
#define VM_BITERATE_6			9600
#define VM_BITERATE_7			19200
#define VM_BITERATE_8			38400
#define VM_BITERATE_9			43000
#define VM_BITERATE_A			56000
#define VM_BITERATE_B			57600
#define VM_BITERATE_C			115200
#define VM_BITERATE_D			230400


/*-- 视频流类型 --*/
enum XMN_STREAM_TYPE{
	XMN_STREAM_HD=0,	 // 高清、主码流
	XMN_STREAM_SD,	 	 // 标清、子码流
};


enum XMN_PLAY_TYPE{
	XM_PLAY_TYPE_LIVE=0,        	//实时播放
	XM_PLAY_TYPE_RECORD,        	//回放
	XM_PLAY_TYPE_LINK_LIVE,         //枪球联动实时播放
	XM_PLAY_TYPE_LINK_RECORD,       //枪球联动回放
};



#endif //__PUWELLTYPEDEF_H__
