"""
LangGraph 节点函数模块
包含工作流中的各个处理节点
"""

import json
from typing import TypedDict, Dict, Any

# 导入 LangChain 相关模块
from langchain_openai import ChatOpenAI

# 导入自定义模块
from config import (
    ARK_API_KEY,
    ARK_BASE_URL,
    ARK_MODEL_ID,
    TEMPERATURE,
    MAX_TOKENS
)
from prompts import TABLE_EVALUATION_PROMPT, PERMISSION_EVALUATION_PROMPT
from feishu_api import extract_bitable_info_with_permissions


# ==================== 状态定义 ====================
class EvaluationState(TypedDict):
    """
    LangGraph 状态定义
    用于在工作流节点之间传递数据
    """
    odata_url: str          # 原始样张飞书链接
    ndata_url: str          # 需要清洁的样张飞书链接
    odata: str             # 原始样张 JSON 字符串（包含 tables 和 permissions）
    ndata: str             # 需要清洁的样张 JSON 字符串（包含 tables 和 permissions）
    table_analysis_result: Dict[str, Any]  # 表结构 LLM 分析结果
    permission_analysis_result: Dict[str, Any]  # 权限 LLM 分析结果
    result: str            # 评测结果：正确/异常
    reason: str            # 原因说明
    skip_llm: bool         # 是否跳过 LLM 节点


# ==================== 节点函数 ====================

def parse_bitable_urls(state: EvaluationState) -> EvaluationState:
    """
    解析飞书链接，获取数据表 metadata 和高级权限信息的节点
    
    Args:
        state: 当前的评测状态，包含 odata_url 和 ndata_url
        
    Returns:
        更新后的评测状态，包含解析后的 odata 和 ndata JSON
    """
    odata_url = state["odata_url"]
    ndata_url = state["ndata_url"]
    
    # 解析原始样张（包含 tables 和 permissions）
    try:
        odata_info = extract_bitable_info_with_permissions(odata_url)
        state["odata"] = json.dumps(odata_info, ensure_ascii=False)
    except Exception as e:
        state["odata"] = json.dumps({"error": f"解析原始样张失败: {str(e)}"}, ensure_ascii=False)
        state["result"] = "处理失败"
        state["reason"] = f"解析原始样张失败: {str(e)}"
        return state
    
    # 解析需要清洁的样张（包含 tables 和 permissions）
    try:
        ndata_info = extract_bitable_info_with_permissions(ndata_url)
        state["ndata"] = json.dumps(ndata_info, ensure_ascii=False)
    except Exception as e:
        state["ndata"] = json.dumps({"error": f"解析需要清洁的样张失败: {str(e)}"}, ensure_ascii=False)
        state["result"] = "处理失败"
        state["reason"] = f"解析需要清洁的样张失败: {str(e)}"
        return state
    
    # 解析成功，重置 result 和 reason
    state["result"] = ""
    state["reason"] = ""
    state["skip_llm"] = False
    state["table_analysis_result"] = {}
    state["permission_analysis_result"] = {}
    
    return state


def normalize_table_data(data: Dict) -> Dict:
    """
    标准化表格数据，移除不稳定的字段（如 field_id），并排序以方便对比
    
    Args:
        data: 原始表格数据
        
    Returns:
        标准化后的表格数据
    """
    normalized = data.copy()
    # 移除 table_id（虽然在 extract_base_token_and_tables 中已经移除了，但这里再处理一次以防万一）
    normalized.pop("table_id", None)
    
    # 处理 fields，移除 field_id
    if "fields" in normalized:
        normalized_fields = []
        for field in normalized["fields"]:
            normalized_field = field.copy()
            normalized_field.pop("field_id", None)
            normalized_fields.append(normalized_field)
        # 对 fields 按 field_name 排序
        normalized_fields.sort(key=lambda x: x.get("field_name", ""))
        normalized["fields"] = normalized_fields
    
    return normalized


def normalize_permission_data(data: Dict) -> Dict:
    """
    标准化权限数据，移除不稳定的字段（如 role_id、table_id），并排序以方便对比
    
    Args:
        data: 原始权限数据
        
    Returns:
        标准化后的权限数据
    """
    normalized = data.copy()
    
    # 处理 roles
    if "roles" in normalized:
        normalized_roles = []
        for role in normalized["roles"]:
            normalized_role = role.copy()
            # 移除 role_id（因为这是不稳定的字段
            normalized_role.pop("role_id", None)
            
            # 处理 table_roles
            if "table_roles" in normalized_role:
                normalized_table_roles = []
                for table_role in normalized_role["table_roles"]:
                    normalized_table_role = table_role.copy()
                    # 移除 table_id（不稳定字段）
                    normalized_table_role.pop("table_id", None)
                    
                    # 处理 field_perm - 按字段名排序（稳定比较）
                    if "field_perm" in normalized_table_role:
                        # 转换为有序字典按 key 排序，保证比较稳定
                        field_perm_dict = normalized_table_role["field_perm"]
                        if isinstance(field_perm_dict, dict):
                            # 按字段名排序后重建
                            sorted_field_perm = dict(sorted(field_perm_dict.items()))
                            normalized_table_role["field_perm"] = sorted_field_perm
                    
                    normalized_table_roles.append(normalized_table_role)
                
                # 对 table_roles 按 table_name 排序（稳定比较）
                normalized_table_roles.sort(key=lambda x: x.get("table_name", ""))
                normalized_role["table_roles"] = normalized_table_roles
            
            # 处理 block_roles
            if "block_roles" in normalized_role:
                normalized_block_roles = []
                for block_role in normalized_role["block_roles"]:
                    normalized_block_role = block_role.copy()
                    # 移除 block_id（不稳定字段）
                    normalized_block_role.pop("block_id", None)
                    normalized_block_roles.append(normalized_block_role)
                
                # 对 block_roles 排序（保证稳定比较）
                normalized_block_roles.sort(key=lambda x: (x.get("block_type", ""), x.get("block_perm", 0)))
                normalized_role["block_roles"] = normalized_block_roles
            
            normalized_roles.append(normalized_role)
        
        # 对 roles 按 role_name 排序
        normalized_roles.sort(key=lambda x: x.get("role_name", ""))
        normalized["roles"] = normalized_roles
    
    return normalized


def compare_tables(state: EvaluationState) -> EvaluationState:
    """
    代码节点：对比 odata 和 ndata 的 tables 和 permissions 是否一致
    
    Args:
        state: 当前的评测状态
        
    Returns:
        更新后的评测状态
    """
    # 如果之前解析失败，直接返回
    if state.get("result") == "处理失败":
        state["skip_llm"] = True
        return state
    
    try:
        import json
        odata = json.loads(state["odata"])
        ndata = json.loads(state["ndata"])
        
        # 对比 tables 部分
        odata_tables = odata.get("tables", [])
        ndata_tables = ndata.get("tables", [])
        
        # 标准化两个 tables 数据
        normalized_odata_tables = [normalize_table_data(t) for t in odata_tables]
        normalized_ndata_tables = [normalize_table_data(t) for t in ndata_tables]
        
        # 对 tables 按 table_name 排序
        normalized_odata_tables.sort(key=lambda x: x.get("table_name", ""))
        normalized_ndata_tables.sort(key=lambda x: x.get("table_name", ""))
        
        tables_match = (normalized_odata_tables == normalized_ndata_tables)
        
        # 对比 permissions 部分
        odata_permissions = odata.get("permissions", {})
        ndata_permissions = ndata.get("permissions", {})
        
        # 标准化两个 permissions 数据
        normalized_odata_permissions = normalize_permission_data(odata_permissions)
        normalized_ndata_permissions = normalize_permission_data(ndata_permissions)
        
        permissions_match = (normalized_odata_permissions == normalized_ndata_permissions)
        
        # 只有当 tables 和 permissions 都一致时，才标记为正确
        if tables_match and permissions_match:
            state["result"] = "正确"
            state["reason"] = "代码节点判断"
            state["skip_llm"] = True
        else:
            state["skip_llm"] = False
            
    except Exception as e:
        state["skip_llm"] = False
    
    return state


def extract_tables_from_json(json_str: str) -> str:
    """
    从完整的 JSON 字符串中提取 tables 部分
    
    Args:
        json_str: 包含 tables 和 permissions 的完整 JSON
        
    Returns:
        只包含 tables 的 JSON 字符串
    """
    try:
        data = json.loads(json_str)
        tables = data.get("tables", [])
        return json.dumps({"tables": tables}, ensure_ascii=False)
    except Exception:
        return json_str


def extract_permissions_from_json(json_str: str) -> str:
    """
    从完整的 JSON 字符串中提取 permissions 部分
    
    Args:
        json_str: 包含 tables 和 permissions 的完整 JSON
        
    Returns:
        只包含 permissions 的 JSON 字符串
    """
    try:
        data = json.loads(json_str)
        permissions = data.get("permissions", {})
        return json.dumps({"permissions": permissions}, ensure_ascii=False)
    except Exception:
        return json_str


def analyze_table_with_llm(state: EvaluationState) -> EvaluationState:
    """
    使用 LLM 进行表结构一致性分析的节点
    
    Args:
        state: 当前的评测状态，包含解析后的 odata 和 ndata
        
    Returns:
        更新后的评测状态，包含表结构分析结果
    """
    # 如果代码节点已经判断完成或之前解析失败，直接返回
    if state.get("result") == "处理失败" or state.get("skip_llm", False):
        return state
    
    llm = ChatOpenAI(
        model=ARK_MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=ARK_API_KEY,
        base_url=ARK_BASE_URL
    )
    
    # 提取 tables 部分
    odata_tables = extract_tables_from_json(state["odata"])
    ndata_tables = extract_tables_from_json(state["ndata"])
    
    # 格式化提示词
    prompt = TABLE_EVALUATION_PROMPT.format(
        odata=odata_tables,
        ndata=ndata_tables
    )
    
    # 调用 LLM
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # 解析 LLM 输出的 JSON
    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx+1]
            result = json.loads(json_str)
        else:
            result = json.loads(content)
    except Exception as e:
        result = {
            "result": "处理失败",
            "reason": f"表结构LLM解析失败: {str(e)}, 原始内容: {content[:100]}"
        }
    
    # 更新状态
    state["table_analysis_result"] = result
    return state


def analyze_permission_with_llm(state: EvaluationState) -> EvaluationState:
    """
    使用 LLM 进行权限一致性分析的节点
    
    Args:
        state: 当前的评测状态，包含解析后的 odata 和 ndata
        
    Returns:
        更新后的评测状态，包含权限分析结果
    """
    # 如果代码节点已经判断完成或之前解析失败，直接返回
    if state.get("result") == "处理失败" or state.get("skip_llm", False):
        return state
    
    llm = ChatOpenAI(
        model=ARK_MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=ARK_API_KEY,
        base_url=ARK_BASE_URL
    )
    
    # 提取 permissions 部分
    odata_permissions = extract_permissions_from_json(state["odata"])
    ndata_permissions = extract_permissions_from_json(state["ndata"])
    
    # 格式化提示词
    prompt = PERMISSION_EVALUATION_PROMPT.format(
        odata=odata_permissions,
        ndata=ndata_permissions
    )
    
    # 调用 LLM
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # 解析 LLM 输出的 JSON
    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx+1]
            result = json.loads(json_str)
        else:
            result = json.loads(content)
    except Exception as e:
        result = {
            "result": "处理失败",
            "reason": f"权限LLM解析失败: {str(e)}, 原始内容: {content[:100]}"
        }
    
    # 更新状态
    state["permission_analysis_result"] = result
    return state


def combine_results(state: EvaluationState) -> EvaluationState:
    """
    合并表结构和权限分析结果的节点
    
    Args:
        state: 当前的评测状态，包含两个分析结果
        
    Returns:
        更新后的评测状态，包含最终结果
    """
    # 如果代码节点已经判断完成或之前解析失败，直接返回
    if state.get("result") == "处理失败" or state.get("skip_llm", False):
        return state
    
    table_result = state.get("table_analysis_result", {})
    permission_result = state.get("permission_analysis_result", {})
    
    table_status = table_result.get("result", "处理失败")
    permission_status = permission_result.get("result", "处理失败")
    
    table_reason = table_result.get("reason", "")
    permission_reason = permission_result.get("reason", "")
    
    # 合并原因
    reasons = []
    if table_status == "异常" and table_reason:
        reasons.append(f"[表结构] {table_reason}")
    if permission_status == "异常" and permission_reason:
        reasons.append(f"[权限] {permission_reason}")
    if table_status == "处理失败" and table_reason:
        reasons.append(f"[表结构] {table_reason}")
    if permission_status == "处理失败" and permission_reason:
        reasons.append(f"[权限] {permission_reason}")
    
    # 判断最终结果 - 严格逻辑
    if table_status == "正确" and permission_status == "正确":
        # 只有当表结构和权限配置都正确时，才标记为正确
        state["result"] = "正确"
        state["reason"] = "表结构和权限配置均一致"
    elif table_status == "处理失败" or permission_status == "处理失败":
        # 只要有一个处理失败，就标记为处理失败
        state["result"] = "处理失败"
        state["reason"] = "; ".join(reasons) if reasons else "处理失败"
    else:
        # 其他情况都标记为异常
        state["result"] = "异常"
        state["reason"] = "; ".join(reasons) if reasons else "存在不一致"
    
    return state
