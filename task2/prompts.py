"""提示模板设计：为中药处方生成任务定制的提示工程"""

import random

def get_common_herbs_for_syndrome(syndrome):
    """为特定证型返回常用药物提示"""
    syndrome_herbs = {
        "气虚血瘀证": "黄芪、党参、当归、川芎、丹参、赤芍",
        "痰瘀互结证": "陈皮、半夏、茯苓、胆南星、瓜蒌、丹参、赤芍",
        "气阴两虚证": "黄芪、党参、麦冬、五味子、当归、白芍",
        "气滞血瘀证": "柴胡、香附、川芎、赤芍、丹参、延胡索",
        "肝阳上亢证": "天麻、钩藤、石决明、菊花、牛膝、白芍",
        "阴虚阳亢证": "生地黄、知母、黄柏、牡丹皮、天麻、钩藤",
        "痰热蕴结证": "黄连、黄芩、栀子、胆南星、半夏、茯苓",
        "痰湿痹阻证": "苍术、白术、茯苓、薏苡仁、威灵仙、秦艽",
        "阳虚水停证": "附子、干姜、肉桂、白术、茯苓、猪苓",
        "肝肾阴虚证": "生地黄、山茱萸、山药、丹皮、白芍、知母"
    }
    
    return syndrome_herbs.get(syndrome, "")

def get_classical_formula_for_disease(disease):
    """为特定疾病返回经典方剂框架"""
    disease_formulas = {
        "胸痹心痛病": "参考方剂：血府逐瘀汤、瓜蒌薤白白酒汤、桂枝茯苓丸",
        "心衰病": "参考方剂：真武汤、生脉散、参芪杞膏",
        "眩晕病": "参考方剂：天麻钩藤饮、镇肝息风汤、补阳还五汤",
        "心悸病": "参考方剂：甘麦大枣汤、温胆汤、归脾汤"
    }
    
    return disease_formulas.get(disease, "")

def get_similar_cases(syndromes, disease, n=2):
    """获取n个相似案例用于少样本学习"""
    # 此处应该从训练数据中检索相似案例，这里简化为几个预设案例
    similar_cases = []
    
    # 预设案例库（实际应用中从训练集中动态选择）
    case_examples = [
        {
            "syndromes": ["气虚血瘀证"],
            "disease": "胸痹心痛病",
            "symptoms": "胸闷胸痛，心悸气短，倦怠乏力，舌暗紫，脉细涩",
            "herbs": "黄芪、党参、当归、川芎、丹参、赤芍、茯苓、桂枝、炙甘草、三七粉、麦冬、白术"
        },
        {
            "syndromes": ["痰热蕴结证"],
            "disease": "胸痹心痛病",
            "symptoms": "胸闷胸痛，心烦气躁，痰多色黄，舌红苔黄腻，脉滑数",
            "herbs": "黄连、黄芩、栀子、瓜蒌、半夏、茯苓、黄芪、丹参、竹茹、陈皮、枳壳、甘草"
        },
        {
            "syndromes": ["阴虚阳亢证"],
            "disease": "眩晕病",
            "symptoms": "头晕目眩，耳鸣健忘，腰膝酸软，五心烦热，舌红少苔，脉细数",
            "herbs": "生地黄、知母、黄柏、天麻、钩藤、菊花、牡丹皮、白芍、龟板、石决明、磁石、茯神"
        },
        {
            "syndromes": ["痰湿痹阻证"],
            "disease": "心悸病",
            "symptoms": "心悸怔忡，胸闷不舒，痰多体重，头重如裹，舌淡胖有齿痕，苔白腻，脉沉滑",
            "herbs": "苍术、白术、茯苓、半夏、陈皮、竹茹、枳实、远志、石菖蒲、炙甘草、生姜、大枣"
        }
    ]
    
    # 选择与当前证型疾病最匹配的案例
    for case in case_examples:
        if any(s in case["syndromes"] for s in syndromes) and case["disease"] == disease:
            similar_cases.append(case)
            if len(similar_cases) >= n:
                break
    
    # 如果没有找到足够的匹配案例，添加其他案例凑数
    while len(similar_cases) < n and case_examples:
        similar_cases.append(random.choice(case_examples))
        
    return similar_cases

def generate_base_prompt(patient_info, herbs_list, predicted_syndromes=None, predicted_disease=None):
    """生成基础提示模板"""
    # 提取患者信息
    gender = patient_info.get('性别', '')
    age = patient_info.get('年龄', '')
    job = patient_info.get('职业', '')
    marriage = patient_info.get('婚姻', '')
    disease_time = patient_info.get('发病节气', '')
    chief_complaint = patient_info.get('主诉', '')
    symptom = patient_info.get('症状', '')
    tcm_examination = patient_info.get('中医望闻切诊', '')
    history = patient_info.get('病史', '')
    physical_examination = patient_info.get('体格检查', '')
    auxiliary_examination = patient_info.get('辅助检查', '')
    
    # 如果没有传入预测结果，则使用真实标签（如果有）
    syndromes = predicted_syndromes or (patient_info.get('证型', '').split('|') if '证型' in patient_info else [])
    disease = predicted_disease or patient_info.get('疾病', '')
    
    # 证型和疾病特定信息
    syndrome_herbs_info = ""
    for syndrome in syndromes:
        common_herbs = get_common_herbs_for_syndrome(syndrome)
        if common_herbs:
            syndrome_herbs_info += f"对于{syndrome}常用药物: {common_herbs}\n"
    
    disease_formula_info = get_classical_formula_for_disease(disease) if disease else ""
    
    # 构建基础提示
    prompt = f"""任务：作为经验丰富的中医师，根据患者[基本信息],[主诉],[症状],[中医望闻切诊],[病史],[体格检查],[辅助检查]等信息，为患者开具合适的中药处方。

# 中医辨证结果
证型判断：{' '.join(syndromes)}
疾病判断：{disease}

# 处方原则
1. 必须严格从下方药物列表中选择，不得添加列表外药物
2. 根据"君臣佐使"原则组方，兼顾疾病和证型特点
3. 药物数量应控制在10-15味之间
4. 输出格式为逗号分隔的药物列表，不包含剂量

{syndrome_herbs_info}
{disease_formula_info}

# 可选药物列表
[草药]: {', '.join(herbs_list)}

[基本信息]:患者性别为{gender},年龄为{age},职业为{job},婚姻为{marriage},发病节气为{disease_time}
[主诉]:{chief_complaint}
[症状]:{symptom}
[中医望闻切诊]:{tcm_examination}
[病史]:{history}
[体格检查]:{physical_examination}
[辅助检查]:{auxiliary_examination}

请根据以上信息，列出推荐的中药处方:"""
    
    return prompt

def generate_few_shot_prompt(patient_info, herbs_list, predicted_syndromes=None, predicted_disease=None):
    """生成少样本学习提示"""
    # 获取基础提示
    base_prompt = generate_base_prompt(patient_info, herbs_list, predicted_syndromes, predicted_disease)
    
    # 获取相似案例
    syndromes = predicted_syndromes or (patient_info.get('证型', '').split('|') if '证型' in patient_info else [])
    disease = predicted_disease or patient_info.get('疾病', '')
    similar_cases = get_similar_cases(syndromes, disease)
    
    # 构建少样本示例
    examples = "\n\n# 类似病例参考\n"
    for i, case in enumerate(similar_cases, 1):
        examples += f"""例{i}:
证型: {', '.join(case['syndromes'])}
疾病: {case['disease']}
症状: {case['symptoms']}
处方: {case['herbs']}
"""
    
    # 插入到基础提示和最终请求之间
    parts = base_prompt.split("请根据以上信息，列出推荐的中药处方:")
    if len(parts) == 2:
        few_shot_prompt = parts[0] + examples + "\n请根据以上信息和参考病例，列出推荐的中药处方:"
    else:
        few_shot_prompt = base_prompt + examples
    
    return few_shot_prompt 