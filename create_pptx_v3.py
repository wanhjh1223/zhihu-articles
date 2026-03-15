#!/usr/bin/env python3
"""
创建美观的PPTX文件 - 焦虑抑郁的产生及原因
使用纯Python和zipfile模块，添加视觉效果
"""

import zipfile
import os
import io
from datetime import datetime
import base64

def create_pptx(output_path):
    """创建一个带有视觉效果的PPTX文件"""
    
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
    <cp:revision>3</cp:revision>
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
        
        # 其他必要文件...
        zf.writestr('ppt/presProps.xml', get_presProps())
        zf.writestr('ppt/viewProps.xml', get_viewProps())
        zf.writestr('ppt/tableStyles.xml', get_tableStyles())
        zf.writestr('ppt/theme/theme1.xml', get_theme())
        zf.writestr('ppt/theme/_rels/theme1.xml.rels', get_theme_rels())
        zf.writestr('ppt/slideMasters/slideMaster1.xml', get_slideMaster())
        zf.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels', get_master_rels())
        zf.writestr('ppt/slideLayouts/slideLayout1.xml', get_slideLayout())
        zf.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels', get_layout_rels())
        
        # 创建12张幻灯片
        slides_data = get_slides_data()
        
        for i, slide_data in enumerate(slides_data, 1):
            is_title_slide = (i == 1)
            slide_xml = create_slide_v2(i, slide_data["title"], slide_data.get("subtitle", ""), slide_data["content"], is_title_slide)
            zf.writestr(f'ppt/slides/slide{i}.xml', slide_xml)
            
            slide_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>'''
            zf.writestr(f'ppt/slides/_rels/slide{i}.xml.rels', slide_rels)
    
    with open(output_path, 'wb') as f:
        f.write(pptx_buffer.getvalue())
    
    print(f"美观PPTX文件已创建: {output_path}")

def get_slides_data():
    """返回所有幻灯片数据"""
    return [
        {
            "title": "焦虑抑郁的产生及原因",
            "subtitle": "心理学与神经科学视角下的病因分析",
            "content": "原生家庭深度解析 | 2026年3月"
        },
        {
            "title": "什么是焦虑抑郁？",
            "content": """焦虑：以紧张、担心和身体变化为特征的情绪

抑郁：情绪低落、兴趣缺失、消极观念的精神状态

焦虑性抑郁：两者常结伴而来
约70%慢性焦虑患者会发展为抑郁症

流行病学数据：
• 全球约18%成年人患有焦虑症
• 全球约3.5亿抑郁症患者  
• 近2/3成年人有童年不良经历(ACE)"""
        },
        {
            "title": "原生家庭：焦虑抑郁的温床",
            "content": """童年创伤的长期影响

童年逆境占焦虑障碍人口可归因风险：32.4%
有过童年创伤的人患病可能性是普通人的2倍
童年创伤可预测地解释了26.2%的情绪障碍

关键研究发现：
• ACE研究：童年不良经历与抑郁风险呈剂量依赖关系
• 近2/3美国成年人经历过至少一次童年不良经历
• 情感忽视的影响往往比身体虐待更深远"""
        },
        {
            "title": "原生家庭行为（一）：控制与压制",
            "content": """专制型父母行为：
• 高要求、低回应，强调服从与纪律
• 使用惩罚维持规范，而非沟通
• 孩子没有参与决策的权力

过度控制与保护：
• 限制孩子接触产生焦虑的情境
• "直升机父母"：盘旋在孩子身上
• 侵入性养育：替孩子完成任务

后果：情绪焦虑、自卑、缺乏探索动力"""
        },
        {
            "title": "原生家庭行为（二）：情感忽视与拒绝",
            "content": """情感忽视：
• 对孩子的需求和感受缺乏回应
• 情感冷漠，回避亲密关系
• 不表达爱、温暖和支持

情感虐待与拒绝：
• 过度批评、贬低、羞辱
• "这不行、那不行"的否定教育

造成的后果：
• 形成"我不够好"的核心信念
• 社交恐惧，害怕被拒绝
• 习惯性讨好他人
• 成年后难以建立健康亲密关系"""
        },
        {
            "title": "原生家庭行为（三）：示范与强化回避",
            "content": """父母的焦虑示范：
• 孩子通过观察学习父母如何应对
• 焦虑父母的控制行为被孩子模仿
• 传递"世界是危险的"信号

强化孩子的回避行为：
• 替害怕社交的孩子点餐、与店员交谈
• 孩子不愿上学时允许其待在家里

造成的后果：
• 孩子学会逃避是解决问题的方法
• 回避行为被自我强化
• 社交技能发展不足
• 不敢走出家门，不敢尝试新事物"""
        },
        {
            "title": "原生家庭行为（四）：创伤性经历",
            "content": """虐待与暴力：
• 身体虐待：打骂、体罚
• 性虐待：任何形式的性侵害
• 家庭暴力：目睹父母冲突

家庭环境不稳定：
• 父母离异或分居
• 频繁搬家、转学
• 经济困难

造成的后果：
• 创伤后应激障碍(PTSD)
• 对人际关系缺乏基本信任
• 高度警觉(hypervigilance)：总是察言观色
• 成年后更容易患抑郁和焦虑障碍"""
        },
        {
            "title": "生物学因素",
            "content": """1. 遗传因素
   焦虑症具有家族聚集性

2. 脑结构异常
   杏仁核：过度活跃，对威胁过度反应
   前额叶皮层：功能不足，调节能力减弱
   海马体：长期压力导致体积缩小

3. 神经递质失衡
   5-羟色胺水平低下
   GABA（刹车机制）功能减弱

4. HPA轴失调
   皮质醇持续升高
   负反馈机制失灵"""
        },
        {
            "title": "心理因素",
            "content": """认知模式问题：
• 灾难化思维：将小事放大为威胁
• 负面思维循环
• 完美主义：对不确定性容忍度低
• "我不够好"的核心自我认知

性格特质：
• 高度敏感、易受伤
• 缺乏自信心
• 回避行为模式

心理动力学因素：
• 潜意识中的冲突和压抑
• 内化的严苛父母声音
• 恐惧型依恋模式"""
        },
        {
            "title": "大脑神经机制",
            "content": """神经回路失调：
• 杏仁核-前额叶环路连接减弱
• 慢性压力改变信号传导

神经可塑性改变：
• 海马体：新生神经元减少
• 前额叶皮层：执行功能下降
• 杏仁核：反应性增强

神经炎症：
• 促炎细胞因子升高
• 小胶质细胞活化

分子机制：
• SIRT1基因表达下调
• CRF（促肾上腺皮质激素释放因子）上调"""
        },
        {
            "title": "病因总结：多因素交互模型",
            "content": """        原生家庭因素
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
生物学因素  心理因素   社会环境因素
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

童年创伤通过影响认知风格、人格特质，
最终影响成年期精神病理学。"""
        },
        {
            "title": "应对与预防建议",
            "content": """觉察创伤：
• 识别症状背后的童年根源
• 理解"现在的恐惧来自过去的经历"

认知重构：
• 识别"我不够好"等不合理信念
• 用积极思维替代灾难化思维

逐步暴露：
• 从小步骤开始面对恐惧
• 建立"我可以应对"的信心

专业帮助：
• 创伤知情治疗（EMDR）
• 认知行为疗法（CBT）

重要提醒：治愈是可能的！
理解根源是改变的第一步。"""
        }
    ]

def create_slide_v2(num, title, subtitle, content, is_title_slide=False):
    """创建单张幻灯片的XML - 带视觉效果"""
    
    shapes_xml = ""
    
    if is_title_slide:
        # 标题幻灯片 - 带渐变背景效果
        shapes_xml = create_title_slide_shapes(title, subtitle, content)
    else:
        # 内容幻灯片
        shapes_xml = create_content_slide_shapes(title, content)
    
    # 背景 - 使用渐变
    bg_xml = get_gradient_background()
    
    slide_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld>
        {bg_xml}
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

def get_gradient_background():
    """获取渐变背景XML"""
    # 使用蓝色渐变背景
    return '''<p:bg>
    <p:bgPr>
        <a:gradFill rotWithShape="1">
            <a:gsLst>
                <a:gs pos="0"><a:srgbClr val="1E3A5F"/></a:gs>
                <a:gs pos="50000"><a:srgbClr val="2E5984"/></a:gs>
                <a:gs pos="100000"><a:srgbClr val="4A90A4"/></a:gs>
            </a:gsLst>
            <a:lin ang="2700000" scaled="1"/>
        </a:gradFill>
        <a:effectLst/>
    </p:bgPr>
</p:bg>'''

def create_title_slide_shapes(title, subtitle, content):
    """创建标题幻灯片的形状"""
    
    # 装饰性形状 - 顶部色条
    top_bar = '''<p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="TopBar"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="0" y="0"/><a:ext cx="9144000" cy="400000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
            <a:solidFill><a:srgbClr val="FF6B6B"/></a:solidFill>
        </p:spPr>
    </p:sp>'''
    
    # 主标题
    title_shape = f'''<p:sp>
        <p:nvSpPr><p:cNvPr id="3" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="457200" y="1500000"/><a:ext cx="8231800" cy="1000000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            <a:p><a:pPr algn="ctr"><a:defRPr sz="5400" b="1"/></a:pPr>
                <a:r><a:rPr lang="zh-CN" sz="5400" b="1"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:rPr><a:t>{escape_xml(title)}</a:t></a:r>
            </a:p>
        </p:txBody>
    </p:sp>'''
    
    # 副标题
    subtitle_shape = f'''<p:sp>
        <p:nvSpPr><p:cNvPr id="4" name="Subtitle"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="457200" y="2600000"/><a:ext cx="8231800" cy="600000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            <a:p><a:pPr algn="ctr"><a:defRPr sz="2800"/></a:pPr>
                <a:r><a:rPr lang="zh-CN" sz="2800"><a:solidFill><a:srgbClr val="E8E8E8"/></a:solidFill></a:rPr><a:t>{escape_xml(subtitle)}</a:t></a:r>
            </a:p>
        </p:txBody>
    </p:sp>'''
    
    # 底部信息
    footer_shape = f'''<p:sp>
        <p:nvSpPr><p:cNvPr id="5" name="Footer"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="457200" y="5500000"/><a:ext cx="8231800" cy="400000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            <a:p><a:pPr algn="ctr"><a:defRPr sz="2000"/></a:pPr>
                <a:r><a:rPr lang="zh-CN" sz="2000"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill></a:rPr><a:t>{escape_xml(content)}</a:t></a:r>
            </a:p>
        </p:txBody>
    </p:sp>'''
    
    # 装饰圆
    circle = '''<p:sp>
        <p:nvSpPr><p:cNvPr id="6" name="Circle"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="8000000" y="4500000"/><a:ext cx="800000" cy="800000"/></a:xfrm>
            <a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>
            <a:solidFill><a:srgbClr val="FF6B6B"><a:alpha val="50000"/></a:srgbClr></a:solidFill>
        </p:spPr>
    </p:sp>'''
    
    return top_bar + title_shape + subtitle_shape + footer_shape + circle

def create_content_slide_shapes(title, content):
    """创建内容幻灯片的形状"""
    
    # 左侧装饰条
    left_bar = '''<p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="LeftBar"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="0" y="0"/><a:ext cx="200000" cy="6858000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
            <a:solidFill><a:srgbClr val="FF6B6B"/></a:solidFill>
        </p:spPr>
    </p:sp>'''
    
    # 标题背景
    title_bg = '''<p:sp>
        <p:nvSpPr><p:cNvPr id="3" name="TitleBg"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="300000" y="300000"/><a:ext cx="8544000" cy="900000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
            <a:solidFill><a:srgbClr val="FFFFFF"><a:alpha val="15000"/></a:srgbClr></a:solidFill>
        </p:spPr>
    </p:sp>'''
    
    # 标题
    title_shape = f'''<p:sp>
        <p:nvSpPr><p:cNvPr id="4" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="457200" y="400000"/><a:ext cx="8231800" cy="700000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            <a:p><a:pPr><a:defRPr sz="4000" b="1"/></a:pPr>
                <a:r><a:rPr lang="zh-CN" sz="4000" b="1"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:rPr><a:t>{escape_xml(title)}</a:t></a:r>
            </a:p>
        </p:txBody>
    </p:sp>'''
    
    # 内容区域背景
    content_bg = '''<p:sp>
        <p:nvSpPr><p:cNvPr id="5" name="ContentBg"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="457200" y="1300000"/><a:ext cx="8231800" cy="5000000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
            <a:solidFill><a:srgbClr val="FFFFFF"><a:alpha val="10000"/></a:srgbClr></a:solidFill>
        </p:spPr>
    </p:sp>'''
    
    # 内容
    content_shape = f'''<p:sp>
        <p:nvSpPr><p:cNvPr id="6" name="Content"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="600000" y="1400000"/><a:ext cx="7800000" cy="4800000"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
            <a:bodyPr/><a:lstStyle/>
            {create_content_paragraphs(content)}
        </p:txBody>
    </p:sp>'''
    
    # 右下角装饰
    deco = '''<p:sp>
        <p:nvSpPr><p:cNvPr id="7" name="Deco"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
            <a:xfrm><a:off x="8500000" y="6000000"/><a:ext cx="500000" cy="500000"/></a:xfrm>
            <a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>
            <a:solidFill><a:srgbClr val="FFD93D"><a:alpha val="60000"/></a:srgbClr></a:solidFill>
        </p:spPr>
    </p:sp>'''
    
    return left_bar + title_bg + title_shape + content_bg + content_shape + deco

def create_content_paragraphs(content):
    """将内容转换为段落XML - 白色文字"""
    paragraphs = []
    for line in content.split('\n'):
        if line.strip():
            escaped = escape_xml(line)
            if escaped.startswith('•') or escaped.startswith('1.') or escaped.startswith('2.') or escaped.startswith('3.') or escaped.startswith('4.') or escaped.startswith('5.') or escaped.startswith('6.'):
                para = f'<a:p><a:pPr marL="360000" indent="-360000"><a:buChar char="●"/></a:pPr><a:r><a:rPr lang="zh-CN" sz="2000"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:rPr><a:t>{escaped[2:].strip() if escaped.startswith("• ") else escaped}</a:t></a:r></a:p>'
            elif escaped.startswith('  '):
                para = f'<a:p><a:pPr marL="720000"/><a:r><a:rPr lang="zh-CN" sz="1800"><a:solidFill><a:srgbClr val="E8E8E8"/></a:solidFill></a:rPr><a:t>{escaped.strip()}</a:t></a:r></a:p>'
            else:
                para = f'<a:p><a:r><a:rPr lang="zh-CN" sz="2000"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:rPr><a:t>{escaped}</a:t></a:r></a:p>'
            paragraphs.append(para)
    return '\n'.join(paragraphs)

def escape_xml(text):
    """转义XML特殊字符"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

# 辅助函数返回XML字符串
def get_presProps():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:showPr loop="0" showAnimation="1" showNarration="1" useTimings="1">
        <p:presentMode>present</p:presentMode>
        <p:showScrlOpt>present</p:showScrlOpt>
    </p:showPr>
</p:presPr>'''

def get_viewProps():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
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

def get_tableStyles():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>'''

def get_theme():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
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

def get_theme_rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''

def get_slideMaster():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld><p:spTree/></p:cSld>
    <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
    <p:sldLayoutIdLst>
        <p:sldLayoutId id="2147483649" r:id="rId1"/>
    </p:sldLayoutIdLst>
</p:sldMaster>'''

def get_master_rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''

def get_slideLayout():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld><p:spTree/></p:cSld>
</p:sldLayout>'''

def get_layout_rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''

if __name__ == '__main__':
    output_path = '/root/.openclaw/workspace/焦虑抑郁产生及原因_美化版.pptx'
    create_pptx(output_path)
    print(f"文件大小: {os.path.getsize(output_path)} bytes")
