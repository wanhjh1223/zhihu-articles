#!/usr/bin/env python3
"""
创建PPTX文件 - 焦虑抑郁的产生及原因（含原生家庭章节）
使用纯Python和zipfile模块
"""

import zipfile
import os
import io
from datetime import datetime

def create_pptx(output_path):
    """创建一个基本的PPTX文件"""
    
    pptx_buffer = io.BytesIO()
    
    with zipfile.ZipFile(pptx_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
    <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
    <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide3.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide4.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide5.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide6.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide7.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide8.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide9.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide10.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide11.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/slides/slide12.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
    <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
    <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
    <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
    <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
    <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
    <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
        zf.writestr('[Content_Types].xml', content_types)
        
        # .rels
        rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
    <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
        zf.writestr('_rels/.rels', rels)
        
        # docProps/core.xml
        core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" 
    xmlns:dc="http://purl.org/dc/elements/1.1/" 
    xmlns:dcterms="http://purl.org/dc/terms/" 
    xmlns:dcmitype="http://purl.org/dc/dcmitype/" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <dc:title>焦虑抑郁的产生及原因</dc:title>
    <dc:subject>心理学与神经科学科普 - 含原生家庭因素</dc:subject>
    <dc:creator>Kimi Claw</dc:creator>
    <cp:lastModifiedBy>Kimi Claw</cp:lastModifiedBy>
    <cp:revision>2</cp:revision>
    <dcterms:created xsi:type="dcterms:W3CDTF">{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}</dcterms:created>
    <dcterms:modified xsi:type="dcterms:W3CDTF">{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}</dcterms:modified>
</cp:coreProperties>'''
        zf.writestr('docProps/core.xml', core)
        
        # docProps/app.xml
        app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
    <Template></Template>
    <TotalTime>0</TotalTime>
    <Pages>12</Pages>
    <Words>800</Words>
    <Characters>5000</Characters>
    <Application>Microsoft Office PowerPoint</Application>
    <DocSecurity>0</DocSecurity>
    <Lines>80</Lines>
    <Paragraphs>30</Paragraphs>
    <ScaleCrop>false</ScaleCrop>
    <Company></Company>
    <LinksUpToDate>false</LinksUpToDate>
    <CharactersWithSpaces>5500</CharactersWithSpaces>
    <SharedDoc>false</SharedDoc>
    <HyperlinksChanged>false</HyperlinksChanged>
    <AppVersion>16.0000</AppVersion>
</Properties>'''
        zf.writestr('docProps/app.xml', app)
        
        # ppt/presentation.xml
        presentation = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" 
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" 
    xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:sldIdLst>
        <p:sldId id="256" r:id="rId2"/>
        <p:sldId id="257" r:id="rId3"/>
        <p:sldId id="258" r:id="rId4"/>
        <p:sldId id="259" r:id="rId5"/>
        <p:sldId id="260" r:id="rId6"/>
        <p:sldId id="261" r:id="rId7"/>
        <p:sldId id="262" r:id="rId8"/>
        <p:sldId id="263" r:id="rId9"/>
        <p:sldId id="264" r:id="rId10"/>
        <p:sldId id="265" r:id="rId11"/>
        <p:sldId id="266" r:id="rId12"/>
        <p:sldId id="267" r:id="rId13"/>
    </p:sldIdLst>
    <p:sldSz cx="9144000" cy="6858000" type="screen4x3"/>
    <p:notesSz cx="6858000" cy="9144000"/>
    <p:defaultTextStyle>
        <a:defPPr><a:defRPr lang="zh-CN"/></a:defPPr>
    </p:defaultTextStyle>
</p:presentation>'''
        zf.writestr('ppt/presentation.xml', presentation)
        
        # ppt/_rels/presentation.xml.rels
        pres_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
    <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
    <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide3.xml"/>
    <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide4.xml"/>
    <Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide5.xml"/>
    <Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide6.xml"/>
    <Relationship Id="rId8" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide7.xml"/>
    <Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide8.xml"/>
    <Relationship Id="rId10" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide9.xml"/>
    <Relationship Id="rId11" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide10.xml"/>
    <Relationship Id="rId12" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide11.xml"/>
    <Relationship Id="rId13" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide12.xml"/>
    <Relationship Id="rId14" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>
    <Relationship Id="rId15" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
    <Relationship Id="rId16" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>
    <Relationship Id="rId17" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
</Relationships>'''
        zf.writestr('ppt/_rels/presentation.xml.rels', pres_rels)
        
        # ppt/presProps.xml
        presProps = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:showPr loop="0" showAnimation="1" showNarration="1" useTimings="1">
        <p:presentMode>present</p:presentMode>
        <p:showScrlOpt>present</p:showScrlOpt>
    </p:showPr>
</p:presPr>'''
        zf.writestr('ppt/presProps.xml', presProps)
        
        # ppt/viewProps.xml
        viewProps = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:normalViewPr showOutlineIcons="1">
        <p:restoredLeft sz="15620"/>
        <p:restoredTop sz="94660"/>
    </p:normalViewPr>
    <p:slideViewPr showOutlineIcons="1">
        <p:cSldViewPr snapToGrid="1" snapToObjects="1">
            <p:cViewPr varScale="1">
                <p:scale><a:sx n="104" d="100"/><a:sy n="104" d="100"/></p:scale>
                <p:origin x="-120" y="-90"/>
            </p:cViewPr>
            <p:guideLst/>
        </p:cSldViewPr>
    </p:slideViewPr>
    <p:notesViewPr>
        <p:cSldViewPr>
            <p:cViewPr>
                <p:scale><a:sx n="100" d="100"/><a:sy n="100" d="100"/></p:scale>
                <p:origin x="0" y="0"/>
            </p:cViewPr>
        </p:cSldViewPr>
    </p:notesViewPr>
    <p:gridSpacing cx="72000" cy="72000"/>
</p:viewPr>'''
        zf.writestr('ppt/viewProps.xml', viewProps)
        
        # ppt/tableStyles.xml
        tableStyles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>'''
        zf.writestr('ppt/tableStyles.xml', tableStyles)
        
        # ppt/theme/theme1.xml
        theme = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
    <a:themeElements>
        <a:clrScheme name="Office">
            <a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>
            <a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>
            <a:dk2><a:srgbClr val="44546A"/></a:dk2>
            <a:lt2><a:srgbClr val="E7E6E6"/></a:lt2>
            <a:accent1><a:srgbClr val="5B9BD5"/></a:accent1>
            <a:accent2><a:srgbClr val="ED7D31"/></a:accent2>
            <a:accent3><a:srgbClr val="A5A5A5"/></a:accent3>
            <a:accent4><a:srgbClr val="FFC000"/></a:accent4>
            <a:accent5><a:srgbClr val="4472C4"/></a:accent5>
            <a:accent6><a:srgbClr val="70AD47"/></a:accent6>
            <a:hlink><a:srgbClr val="0563C1"/></a:hlink>
            <a:folHlink><a:srgbClr val="954F72"/></a:folHlink>
        </a:clrScheme>
        <a:fontScheme name="Office">
            <a:majorFont><a:latin typeface="Calibri Light"/><a:ea typeface="微软雅黑"/></a:majorFont>
            <a:minorFont><a:latin typeface="Calibri"/><a:ea typeface="微软雅黑"/></a:minorFont>
        </a:fontScheme>
        <a:fmtScheme name="Office">
            <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
            <a:lnStyleLst><a:ln w="9525" cap="flat" cmpd="sng"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
            <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
            <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
        </a:fmtScheme>
    </a:themeElements>
</a:theme>'''
        zf.writestr('ppt/theme/theme1.xml', theme)
        
        # ppt/theme/_rels/theme1.xml.rels
        theme_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''
        zf.writestr('ppt/theme/_rels/theme1.xml.rels', theme_rels)
        
        # ppt/slideMasters/slideMaster1.xml
        slideMaster = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld><p:spTree/></p:cSld>
    <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
    <p:sldLayoutIdLst>
        <p:sldLayoutId id="2147483649" r:id="rId1"/>
    </p:sldLayoutIdLst>
</p:sldMaster>'''
        zf.writestr('ppt/slideMasters/slideMaster1.xml', slideMaster)
        
        # ppt/slideMasters/_rels/slideMaster1.xml.rels
        master_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''
        zf.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels', master_rels)
        
        # ppt/slideLayouts/slideLayout1.xml
        slideLayout = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld><p:spTree/></p:cSld>
</p:sldLayout>'''
        zf.writestr('ppt/slideLayouts/slideLayout1.xml', slideLayout)
        
        # ppt/slideLayouts/_rels/slideLayout1.xml.rels
        layout_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''
        zf.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels', layout_rels)
        
        # 创建12张幻灯片
        slides_content = [
            # Slide 1: 封面
            {
                "title": "焦虑抑郁的产生及原因",
                "subtitle": "心理学与神经科学视角下的病因分析",
                "content": "含原生家庭因素深度解析 | 2026年3月"
            },
            # Slide 2: 概述
            {
                "title": "什么是焦虑抑郁？",
                "content": """• 焦虑：以紧张、担心的想法和血压升高等身体变化为特征的情绪

• 抑郁：情绪低落、兴趣缺失、消极观念，甚至伴有自杀倾向的精神状态

• 焦虑性抑郁：两者常结伴而来，约70%慢性焦虑患者会发展为抑郁症

流行病学数据：
• 全球约18%成年人患有焦虑症
• 全球约3.5亿抑郁症患者
• 脑相关疾病占所有疾病社会负担的28%
• 近2/3美国成年人经历过至少一次童年不良经历(ACE)"""
            },
            # Slide 3: 原生家庭概述
            {
                "title": "原生家庭：焦虑抑郁的温床",
                "content": """童年创伤的长期影响：
• 童年创伤不会随时间消失，而是像湿水泥中的脚印一样长期存在
• 童年逆境占焦虑障碍人口可归因风险比例的32.4%
• 有过童年创伤的人患抑郁障碍的可能性是普通人的2倍
• 童年创伤可预测地解释了26.2%的情绪障碍

关键研究发现：
• ACE研究：童年创伤与重度抑郁的发展存在强烈关联
• 剂量依赖关系：经历越多童年逆境，患病风险越高
• 约70%慢性焦虑患者会发展为抑郁症
• 情感忽视的影响往往比身体虐待更深远"""
            },
            # Slide 4: 原生家庭行为1
            {
                "title": "原生家庭行为（一）：控制与压制",
                "content": """专制型/独裁型父母行为：
• 高要求、低回应，强调服从与纪律
• 使用惩罚维持规范，而非沟通
• 很少关注孩子的意见和情感
• 孩子没有参与决策的权力

过度控制与过度保护：
• 限制孩子接触产生焦虑的情境
• "直升机父母"：盘旋在孩子身上
• 侵入性养育：替孩子完成任务
• 导致孩子无法发展应对技能

造成的后果：
• 情绪焦虑、封闭自己、不快乐、自卑
• 缺乏社交技能，容易出现敌意和受挫感
• 男孩：充满愤怒，有反社会行为倾向
• 女孩：依赖性很强，缺乏探索和挑战动力"""
            },
            # Slide 5: 原生家庭行为2
            {
                "title": "原生家庭行为（二）：情感忽视与拒绝",
                "content": """情感忽视：
• 对孩子的需求和感受缺乏回应
• 情感冷漠，回避亲密关系
• 只关注物质需求，忽视情感需求
• 不表达爱、温暖和支持

情感虐待与拒绝：
• 过度批评、贬低、羞辱
• "这不行、那不行"的否定教育
• 做错事不给饭吃或打骂
• 让孩子觉得"我不值得被爱"

造成的后果：
• 形成"我不够好""我不配被爱"的核心信念
• 低自尊、自我价值感低下
• 社交恐惧，害怕被拒绝和嘲笑
• 不敢表达真实情绪，习惯性讨好他人
• 成年后难以建立健康亲密关系"""
            },
            # Slide 6: 原生家庭行为3
            {
                "title": "原生家庭行为（三）：示范与强化回避",
                "content": """父母的焦虑示范：
• 孩子通过观察学习父母如何应对世界
• 焦虑父母的控制行为被孩子模仿
• 父母表现出对社交的恐惧和回避
• 传递"世界是危险的"信号

强化孩子的回避行为：
• 替害怕社交的孩子点餐、与店员交谈
• 孩子不愿上学时允许其待在家里
• 避免让孩子面对任何可能引起焦虑的情境
• 用"一切都会好的"过度安慰而不面对问题

造成的后果：
• 孩子学会逃避是解决问题的方法
• 回避行为被自我强化，形成恶性循环
• 社交技能发展不足
• 面对新情境时不知所措，产生恐惧
• 不敢走出家门，不敢尝试新事物"""
            },
            # Slide 7: 原生家庭行为4
            {
                "title": "原生家庭行为（四）：创伤性经历",
                "content": """虐待与暴力：
• 身体虐待：打骂、体罚
• 性虐待：任何形式的性侵害
• 家庭暴力：目睹父母冲突或打架
• 遭受欺凌、嘲笑、被取笑

家庭环境的不稳定：
• 父母离异或分居
• 亲人离世
• 频繁搬家、转学
• 经济困难、家庭破产
• 父母长期争吵、冷战

依恋创伤：
• 早期与主要照顾者分离
• 照顾者不可预测、时而热情时而冷漠
• 形成不安全依恋模式

造成的后果：
• 创伤后应激障碍(PTSD)
• 对人际关系缺乏基本信任
• 恐惧被抛弃或恐惧被控制
• 高度警觉(hypervigilance)：总是"察言观色"
• 成年后更容易患抑郁和焦虑障碍"""
            },
            # Slide 8: 生物学因素
            {
                "title": "生物学因素",
                "content": """1. 遗传因素
   • 焦虑症具有一定的家族聚集性
   • 特定基因变异（如5-HTTLPR基因）影响大脑应激反应

2. 脑结构异常
   • 杏仁核：过度活跃，对威胁过度反应
   • 前额叶皮层：功能不足，调节能力减弱
   • 海马体：长期压力导致体积缩小

3. 神经递质失衡
   • 5-羟色胺水平低下
   • GABA（刹车机制）功能减弱
   • 去甲肾上腺素失调

4. HPA轴失调
   • 皮质醇持续升高
   • 负反馈机制失灵"""
            },
            # Slide 9: 心理因素
            {
                "title": "心理因素",
                "content": """认知模式问题：
• 灾难化思维：将小事放大为威胁
• 负面思维循环：过度关注威胁信号
• 完美主义：对不确定性容忍度低
• "我不够好"的核心自我认知

性格特质：
• 高度敏感、易受伤
• 缺乏自信心
• 悲观倾向
• 回避行为模式

心理动力学因素：
• 潜意识中的冲突和压抑
• 早期未满足的需求
• 内化的严苛父母声音
• 恐惧型依恋模式"""
            },
            # Slide 10: 大脑神经机制
            {
                "title": "大脑神经机制",
                "content": """神经回路失调：
• 杏仁核-前额叶环路连接减弱
• 慢性压力改变信号传导

神经可塑性改变：
• 海马体：新生神经元减少，体积缩小
• 前额叶皮层：执行功能下降
• 杏仁核：反应性增强

神经炎症：
• 促炎细胞因子升高
• 小胶质细胞活化

分子机制：
• SIRT1基因表达下调
• CRF（促肾上腺皮质激素释放因子）上调
• 肝-脑轴：肝脏LCN2释放增加

The Body Keeps the Score：
创伤不仅存储在记忆中，还生物性嵌入身体"""
            },
            # Slide 11: 病因总结
            {
                "title": "病因总结：多因素交互模型",
                "content": """        原生家庭因素
    （童年创伤、养育方式）
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
生物学因素  心理因素   社会环境因素
（遗传易感性）（认知模式）（生活压力）
    │         │         │
    └─────────┼─────────┘
              ↓
    大脑神经回路失调 + 神经炎症
              ↓
         焦虑抑郁症状
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
害怕出门/回避社交    不敢尝试新事物
（场所恐惧症）       （习得性无助）

童年创伤通过影响认知风格、人格特质，
最终影响成年期精神病理学"""
            },
            # Slide 12: 应对与预防
            {
                "title": "应对与预防建议",
                "content": """觉察创伤：
• 识别症状背后的童年根源
• 理解"现在的恐惧来自过去的经历"
• 在关系中治愈：建立新的安全依恋

认知重构：
• 识别"我不够好"等不合理信念
• 用积极思维替代灾难化思维
• 接受"父母不完美，但我可以不同"

逐步暴露：
• 从小步骤开始面对恐惧
• 庆祝每一个小胜利
• 建立"我可以应对"的信心

专业帮助：
• 创伤知情治疗（EMDR、躯体体验疗法）
• 认知行为疗法（CBT）
• 必要时药物治疗

重要提醒：治愈是可能的！
理解根源是改变的第一步"""
            }
        ]
        
        for i, slide_data in enumerate(slides_content, 1):
            slide_xml = create_slide(i, slide_data["title"], slide_data.get("subtitle", ""), slide_data["content"])
            zf.writestr(f'ppt/slides/slide{i}.xml', slide_xml)
            
            slide_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>'''
            zf.writestr(f'ppt/slides/_rels/slide{i}.xml.rels', slide_rels)
    
    with open(output_path, 'wb') as f:
        f.write(pptx_buffer.getvalue())
    
    print(f"PPTX文件已创建: {output_path}")

def create_slide(num, title, subtitle, content):
    """创建单张幻灯片的XML"""
    
    shapes_xml = ""
    
    title_shape = f'''<p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="457200" y="457200"/><a:ext cx="8231800" cy="800000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            <a:p><a:pPr><a:defRPr sz="4000" b="1"/></a:pPr>
                <a:r><a:rPr lang="zh-CN" sz="4000" b="1"/><a:t>{escape_xml(title)}</a:t></a:r>
            </a:p>
        </p:txBody>
    </p:sp>'''
    shapes_xml += title_shape
    
    if subtitle:
        subtitle_shape = f'''<p:sp>
            <p:nvSpPr><p:cNvPr id="3" name="Subtitle"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
            <p:spPr>
                <a:xfrm><a:off x="457200" y="1300000"/><a:ext cx="8231800" cy="400000"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
            </p:spPr>
            <p:txBody>
                <a:bodyPr/><a:lstStyle/>
                <a:p><a:pPr><a:defRPr sz="2400"/></a:pPr>
                    <a:r><a:rPr lang="zh-CN" sz="2400"/><a:t>{escape_xml(subtitle)}</a:t></a:r>
                </a:p>
            </p:txBody>
        </p:sp>'''
        shapes_xml += subtitle_shape
        content_y = "1800000"
    else:
        content_y = "1300000"
    
    content_shape = f'''<p:sp>
        <p:nvSpPr><p:cNvPr id="4" name="Content"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="457200" y="{content_y}"/><a:ext cx="8231800" cy="4800000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            {create_content_paragraphs(content)}
        </p:txBody>
    </p:sp>'''
    shapes_xml += content_shape
    
    slide_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld>
        <p:bg>
            <p:bgPr>
                <a:solidFill><a:srgbClr val="F8F9FA"/></a:solidFill>
                <a:effectLst/>
            </p:bgPr>
        </p:bg>
        <p:spTree>
            <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
            <p:grpSpPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>
            </p:grpSpPr>
            {shapes_xml}
        </p:spTree>
    </p:cSld>
    <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''
    
    return slide_xml

def create_content_paragraphs(content):
    """将内容转换为段落XML"""
    paragraphs = []
    for line in content.split('\n'):
        if line.strip():
            escaped = escape_xml(line)
            if escaped.startswith('•') or escaped.startswith('1.') or escaped.startswith('2.') or escaped.startswith('3.') or escaped.startswith('4.'):
                para = f'<a:p><a:pPr marL="360000" indent="-360000"><a:buChar char="•"/></a:pPr><a:r><a:rPr lang="zh-CN" sz="1800"/><a:t>{escaped[2:].strip() if escaped.startswith("• ") else escaped}</a:t></a:r></a:p>'
            else:
                para = f'<a:p><a:r><a:rPr lang="zh-CN" sz="1800"/><a:t>{escaped}</a:t></a:r></a:p>'
            paragraphs.append(para)
    return '\n'.join(paragraphs)

def escape_xml(text):
    """转义XML特殊字符"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

if __name__ == '__main__':
    output_path = '/root/.openclaw/workspace/焦虑抑郁产生及原因.pptx'
    create_pptx(output_path)
    print(f"文件大小: {os.path.getsize(output_path)} bytes")
