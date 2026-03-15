#!/usr/bin/env python3
"""
创建PPTX文件 - 焦虑抑郁产生及原因
使用纯Python和zipfile模块（不依赖python-pptx）
PPTX文件实际上是一个ZIP文件，包含XML文件
"""

import zipfile
import os
import io
from datetime import datetime

def create_pptx(output_path):
    """创建一个基本的PPTX文件"""
    
    # 创建内存中的文件
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
    <dc:subject>心理学与神经科学科普</dc:subject>
    <dc:creator>Kimi Claw</dc:creator>
    <cp:lastModifiedBy>Kimi Claw</cp:lastModifiedBy>
    <cp:revision>1</cp:revision>
    <dcterms:created xsi:type="dcterms:W3CDTF">{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}</dcterms:created>
    <dcterms:modified xsi:type="dcterms:W3CDTF">{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}</dcterms:modified>
</cp:coreProperties>'''
        zf.writestr('docProps/core.xml', core)
        
        # docProps/app.xml
        app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
    <Template></Template>
    <TotalTime>0</TotalTime>
    <Pages>8</Pages>
    <Words>500</Words>
    <Characters>3000</Characters>
    <Application>Microsoft Office PowerPoint</Application>
    <DocSecurity>0</DocSecurity>
    <Lines>50</Lines>
    <Paragraphs>20</Paragraphs>
    <ScaleCrop>false</ScaleCrop>
    <Company></Company>
    <LinksUpToDate>false</LinksUpToDate>
    <CharactersWithSpaces>3500</CharactersWithSpaces>
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
    <Relationship Id="rId10" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>
    <Relationship Id="rId11" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
    <Relationship Id="rId12" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>
    <Relationship Id="rId13" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
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
        
        # ppt/theme/theme1.xml (简化版)
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
        
        # ppt/slideMasters/slideMaster1.xml (简化版)
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
        
        # 创建8张幻灯片
        slides_content = [
            # Slide 1: 封面
            {
                "title": "焦虑抑郁的产生及原因",
                "subtitle": "心理学与神经科学视角下的病因分析",
                "content": "科普讲座 | 2026年3月"
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
• 脑相关疾病占所有疾病社会负担的28%"""
            },
            # Slide 3: 生物学因素
            {
                "title": "产生原因一：生物学因素",
                "content": """1. 遗传因素
   • 家族聚集性明显
   • 特定基因变异影响应激反应

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
            # Slide 4: 心理因素
            {
                "title": "产生原因二：心理因素",
                "content": """认知模式问题：
• 灾难化思维：将小事放大为威胁
• 负面思维循环：过度关注威胁信号
• 完美主义：对不确定性容忍度低

性格特质：
• 高度敏感
• 缺乏自信心
• 悲观倾向
• 回避行为

心理动力学因素：
• 潜意识中的冲突和压抑
• 未解决的心理动力"""
            },
            # Slide 5: 社会环境因素
            {
                "title": "产生原因三：社会环境因素",
                "content": """童年经历：
• 早期创伤（虐待、忽视）
• 缺乏安全感
• 成年后患病风险高出2-4倍

重大生活事件：
• 亲人离世、失业、关系破裂

慢性压力：
• 工作超负荷、经济困难
• 长期社会隔离

现代社会因素：
• 快节奏生活、信息过载
• 社交媒体比较压力
• 对成功的过高期望"""
            },
            # Slide 6: 神经机制
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
• 肝-脑轴：肝脏LCN2释放增加"""
            },
            # Slide 7: 总结
            {
                "title": "病因总结",
                "content": """┌─────────────────────────────────────────────┐
│            焦虑抑郁病因网络                   │
├──────────────┬──────────────┬───────────────┤
│  生物学因素   │  心理因素    │  社会环境因素  │
├──────────────┼──────────────┼───────────────┤
│ • 遗传因素   │ • 认知模式   │ • 童年创伤     │
│ • 脑结构异常 │ • 性格特质   │ • 重大事件     │
│ • 神经递质   │ • 思维模式   │ • 慢性压力     │
│ • HPA轴失调  │ • 应对能力   │ • 社会孤立     │
└──────────────┴──────────────┴───────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│    大脑神经回路失调 + 神经炎症               │
└─────────────────────────────────────────────┘
                    ↓
           焦虑抑郁症状"""
            },
            # Slide 8: 应对建议
            {
                "title": "应对与预防建议",
                "content": """认知重构：
• 识别不合理信念
• 用积极思维替代灾难化思维

生活方式调整：
• 规律作息，维护生物钟
• 适度运动（跑步、游泳）
• 保证充足睡眠

社会支持：
• 与亲友倾诉交流
• 构建支持网络

专业帮助：
• 症状持续>2周且影响生活→就医
• 心理咨询（认知行为疗法）
• 必要时药物治疗

重要提醒：焦虑抑郁是复杂的多因素疾病，
不是个人软弱的表现！"""
            }
        ]
        
        for i, slide_data in enumerate(slides_content, 1):
            slide_xml = create_slide(i, slide_data["title"], slide_data.get("subtitle", ""), slide_data["content"])
            zf.writestr(f'ppt/slides/slide{i}.xml', slide_xml)
            
            # 每张幻灯片的rels
            slide_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>'''
            zf.writestr(f'ppt/slides/_rels/slide{i}.xml.rels', slide_rels)
    
    # 保存文件
    with open(output_path, 'wb') as f:
        f.write(pptx_buffer.getvalue())
    
    print(f"PPTX文件已创建: {output_path}")

def create_slide(num, title, subtitle, content):
    """创建单张幻灯片的XML"""
    
    # 构建形状XML
    shapes_xml = ""
    
    # 标题形状
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
    
    # 副标题（如果有）
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
    
    # 内容形状
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
            # 检查是否是列表项
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
