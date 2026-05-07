import os
import csv
from dataclasses import dataclass
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from langgraph.graph import StateGraph, END
from config import (
    INPUT_CSV,
    OUTPUT_CSV,
    MAX_WORKERS
)
from evaluation_nodes import (
    EvaluationState,
    parse_bitable_urls,
    compare_tables,
    analyze_table_with_llm,
    analyze_permission_with_llm,
    combine_results
)


@dataclass
class EvaluationResult:
    row_id: int
    odata_base_token: str
    ndata_base_token: str
    odata_tables: str
    ndata_tables: str
    odata_permissions: str
    ndata_permissions: str
    result: str
    reason: str


def create_evaluation_graph():
    graph = StateGraph(EvaluationState)
    graph.add_node("parse", parse_bitable_urls)
    graph.add_node("compare", compare_tables)
    graph.add_node("analyze_table", analyze_table_with_llm)
    graph.add_node("analyze_permission", analyze_permission_with_llm)
    graph.add_node("combine", combine_results)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "compare")
    graph.add_edge("compare", "analyze_table")
    graph.add_edge("analyze_table", "analyze_permission")
    graph.add_edge("analyze_permission", "combine")
    graph.add_edge("combine", END)
    return graph.compile()


def extract_base_token_tables_and_permissions(json_str: str) -> tuple[str, str, str]:
    try:
        import json
        data = json.loads(json_str)
        base_token = data.get("base_token", "")
        tables = data.get("tables", [])
        permissions = data.get("permissions", {})
        
        processed_tables = []
        for table in tables:
            processed_table = {k: v for k, v in table.items() if k != "table_id"}
            if "fields" in processed_table:
                processed_fields = []
                for field in processed_table["fields"]:
                    processed_field = {k: v for k, v in field.items() if k != "field_id"}
                    processed_fields.append(processed_field)
                processed_table["fields"] = processed_fields
            processed_tables.append(processed_table)
        
        processed_permissions = permissions.copy()
        if "roles" in processed_permissions:
            processed_roles = []
            for role in processed_permissions["roles"]:
                processed_role = role.copy()
                processed_role.pop("role_id", None)
                
                if "table_roles" in processed_role:
                    processed_table_roles = []
                    for table_role in processed_role["table_roles"]:
                        processed_table_role = table_role.copy()
                        processed_table_role.pop("table_id", None)
                        if "field_perm" in processed_table_role:
                            field_perm_dict = processed_table_role["field_perm"]
                            if isinstance(field_perm_dict, dict):
                                sorted_field_perm = dict(sorted(field_perm_dict.items()))
                                processed_table_role["field_perm"] = sorted_field_perm
                        processed_table_roles.append(processed_table_role)
                    processed_table_roles.sort(key=lambda x: x.get("table_name", ""))
                    processed_role["table_roles"] = processed_table_roles
                
                if "block_roles" in processed_role:
                    processed_block_roles = []
                    for block_role in processed_role["block_roles"]:
                        processed_block_role = block_role.copy()
                        processed_block_role.pop("block_id", None)
                        processed_block_roles.append(processed_block_role)
                    processed_block_roles.sort(key=lambda x: (x.get("block_type", ""), x.get("block_perm", 0)))
                    processed_role["block_roles"] = processed_block_roles
                
                processed_roles.append(processed_role)
            
            processed_roles.sort(key=lambda x: x.get("role_name", ""))
            processed_permissions["roles"] = processed_roles
        
        tables_json_str = json.dumps(processed_tables, ensure_ascii=False)
        permissions_json_str = json.dumps(processed_permissions, ensure_ascii=False)
        return base_token, tables_json_str, permissions_json_str
    except Exception:
        return "", json_str, ""


def evaluate_single_row(row_id: int, odata_url: str, ndata_url: str) -> EvaluationResult:
    try:
        graph = create_evaluation_graph()
        initial_state = {
            "odata_url": odata_url,
            "ndata_url": ndata_url,
            "odata": "",
            "ndata": "",
            "table_analysis_result": {},
            "permission_analysis_result": {},
            "result": "处理失败",
            "reason": "",
            "skip_llm": False
        }
        result = graph.invoke(initial_state)
        odata_base_token, odata_tables, odata_permissions = extract_base_token_tables_and_permissions(result.get("odata", ""))
        ndata_base_token, ndata_tables, ndata_permissions = extract_base_token_tables_and_permissions(result.get("ndata", ""))
        return EvaluationResult(
            row_id=row_id,
            odata_base_token=odata_base_token,
            ndata_base_token=ndata_base_token,
            odata_tables=odata_tables,
            ndata_tables=ndata_tables,
            odata_permissions=odata_permissions,
            ndata_permissions=ndata_permissions,
            result=result["result"],
            reason=result["reason"]
        )
    except Exception as e:
        return EvaluationResult(
            row_id=row_id,
            odata_base_token="",
            ndata_base_token="",
            odata_tables=f"URL: {odata_url}",
            ndata_tables=f"URL: {ndata_url}",
            odata_permissions="",
            ndata_permissions="",
            result="处理失败",
            reason=str(e)
        )


# ==================== 主函数 ====================
def main():
    """
    主函数：读取 CSV 输入（飞书链接），批量解析并评测，输出结果
    """
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_CSV):
        print(f"错误：输入文件 {INPUT_CSV} 不存在")
        return
    
    print(f"开始评测，并发数: {MAX_WORKERS}...")
    
    # 尝试多种编码读取输入 CSV
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    rows_data = []
    
    for encoding in encodings:
        try:
            with open(INPUT_CSV, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                
                print(f"使用编码: {encoding}")
                print(f"CSV列名: {reader.fieldnames}")
                
                # 读取所有行
                for row in reader:
                    rows_data.append(row)
                
                break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"编码 {encoding} 读取失败: {e}")
            continue
    
    if not rows_data:
        print(f"错误：无法读取文件，请检查文件编码")
        return
    
    # 检查必需的列是否存在
    first_row = rows_data[0] if rows_data else {}
    required_fields = ["原始样张", "需要清洁的样张"]
    for field in required_fields:
        if field not in first_row:
            raise RuntimeError(f"CSV中必须包含 {field} 列")
    
    # 预处理所有行数据（提取飞书链接）
    tasks = []
    for idx, row in enumerate(rows_data, 1):
        odata_url = row.get("原始样张", "").strip()    # 原始样张飞书链接
        ndata_url = row.get("需要清洁的样张", "").strip()  # 需要清洁的样张飞书链接
        
        if not odata_url or not ndata_url:
            print(f"警告：第 {idx} 行缺少飞书链接，跳过")
            continue
        
        tasks.append((idx, odata_url, ndata_url))
    
    print(f"共 {len(tasks)} 条数据待处理")
    
    # 并发处理
    results_dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(evaluate_single_row, idx, odata, ndata): idx for idx, odata, ndata in tasks}
        
        # 收集结果
        completed = 0
        for future in as_completed(futures):
            row_id = futures[future]
            try:
                eval_result = future.result()
                results_dict[row_id] = eval_result
                completed += 1
                
                print(f"[{completed}/{len(tasks)}] 第 {row_id} 条数据完成，结果: {eval_result.result}")
                if eval_result.reason:
                    print(f"  原因: {eval_result.reason}")
            except Exception as e:
                print(f"第 {row_id} 条数据异常: {e}")
    
    # 按原始顺序整理结果
    sorted_results = []
    for idx in range(1, len(tasks) + 1):
        eval_result = results_dict.get(idx)
        if eval_result:
            sorted_results.append({
                "row_id": eval_result.row_id,
                "odata_base_token": eval_result.odata_base_token,
                "ndata_base_token": eval_result.ndata_base_token,
                "odata_tables": eval_result.odata_tables,
                "ndata_tables": eval_result.ndata_tables,
                "odata_permissions": eval_result.odata_permissions,
                "ndata_permissions": eval_result.ndata_permissions,
                "result": eval_result.result,
                "reason": eval_result.reason
            })
    
    # 定义输出 CSV 的列
    fieldnames = ["row_id", "odata_base_token", "ndata_base_token", "odata_tables", "ndata_tables", "odata_permissions", "ndata_permissions", "result", "reason"]
    
    # 写入输出 CSV（使用 UTF-8-sig 编码确保 Excel 能正常打开）
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_results:
            writer.writerow(row)
    
    print(f"\n评测完成！结果已保存到: {OUTPUT_CSV}")
    
    # 统计结果
    total = len(sorted_results)
    correct = sum(1 for r in sorted_results if r["result"] == "正确")
    abnormal = sum(1 for r in sorted_results if r["result"] == "异常")
    failed = sum(1 for r in sorted_results if r["result"] == "处理失败")
    
    print(f"\n统计:")
    print(f"  总数据量: {total}")
    print(f"  正确: {correct}")
    print(f"  异常: {abnormal}")
    print(f"  处理失败: {failed}")


if __name__ == "__main__":
    main()
