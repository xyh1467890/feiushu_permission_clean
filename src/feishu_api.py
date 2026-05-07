import os
import json
import time
import requests
from urllib.parse import urlparse

FEISHU_USER_ACCESS_TOKEN = os.getenv("FEISHU_USER_ACCESS_TOKEN")

if not FEISHU_USER_ACCESS_TOKEN:
    raise RuntimeError("请先在 .env 文件中配置 FEISHU_USER_ACCESS_TOKEN")

FIELD_TYPE_MAPPING = {
    1: "文本", 2: "数字", 3: "单选", 4: "多选", 5: "日期",
    7: "复选框", 11: "人员", 13: "电话号码", 15: "超链接",
    17: "附件", 18: "单项关联", 19: "查找引用", 20: "公式",
    21: "双向关联", 22: "地理位置", 23: "群组",
    1001: "创建时间", 1002: "最后更新时间", 1003: "创建人",
    1004: "修改人", 1005: "自动编号", 3001: "按钮"
}


def get_feishu_headers():
    return {
        "Authorization": f"Bearer {FEISHU_USER_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }


def parse_app_id(feishu_url):
    parsed = urlparse(feishu_url)
    parts = parsed.path.strip("/").split("/")

    if len(parts) < 2 or parts[0] != "base":
        raise ValueError("不是有效的飞书多维表格链接")

    return parts[1]


def feishu_get(url, headers, params=None):
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    data = resp.json()

    if data.get("code") != 0:
        raise RuntimeError(f"接口请求失败：{data}")

    return data.get("data", {})


def get_all_pages(url, headers, params=None):
    params = params or {}
    params = dict(params)
    params.setdefault("page_size", 100)

    items = []
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token

        data = feishu_get(url, headers, params=params)
        items.extend(data.get("items", []))

        if not data.get("has_more"):
            break

        page_token = data.get("page_token")
        time.sleep(0.2)

    return items


def extract_bitable_info(feishu_url, headers=None):
    if headers is None:
        headers = get_feishu_headers()
        
    base_token = parse_app_id(feishu_url)

    tables_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables"
    tables = get_all_pages(tables_url, headers)

    tables_info = []

    for table in tables:
        table_id = table.get("table_id")
        table_name = table.get("table_name") or table.get("name")

        fields_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables/{table_id}/fields"
        fields = get_all_pages(fields_url, headers)

        records_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables/{table_id}/records"
        records = get_all_pages(records_url, headers)

        views_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables/{table_id}/views"
        views = get_all_pages(views_url, headers)

        fields_list = []
        for field in fields:
            field_type_code = field.get("field_type") or field.get("type")
            field_info = {
                "field_id": field.get("field_id"),
                "field_name": field.get("field_name"),
                "field_type_code": field_type_code,
                "field_type_name": FIELD_TYPE_MAPPING.get(field_type_code, f"未知类型({field_type_code})"),
                "is_primary": field.get("is_primary", False)
            }
            if field_type_code in [3, 4]:
                property_data = field.get("property", {})
                if property_data and "options" in property_data:
                    options = []
                    for opt in property_data["options"]:
                        options.append(opt.get("name"))
                    field_info["options"] = options
            fields_list.append(field_info)

        tables_info.append({
            "table_id": table_id,
            "table_name": table_name,
            "record_count": len(records),
            "view_count": len(views),
            "fields": fields_list
        })

    return {
        "base_token": base_token,
        "tables": tables_info
    }


def get_advanced_permissions(base_token, headers=None):
    if headers is None:
        headers = get_feishu_headers()
    
    roles_url = f"https://open.feishu.cn/open-apis/base/v2/apps/{base_token}/roles"
    roles = get_all_pages(roles_url, headers)
    return {"roles": roles}


def extract_bitable_info_with_permissions(feishu_url, headers=None):
    if headers is None:
        headers = get_feishu_headers()
        
    base_token = parse_app_id(feishu_url)

    tables_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables"
    tables = get_all_pages(tables_url, headers)

    tables_info = []

    for table in tables:
        table_id = table.get("table_id")
        table_name = table.get("table_name") or table.get("name")

        fields_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables/{table_id}/fields"
        fields = get_all_pages(fields_url, headers)

        records_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables/{table_id}/records"
        records = get_all_pages(records_url, headers)

        views_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables/{table_id}/views"
        views = get_all_pages(views_url, headers)

        fields_list = []
        for field in fields:
            field_type_code = field.get("field_type") or field.get("type")
            field_info = {
                "field_id": field.get("field_id"),
                "field_name": field.get("field_name"),
                "field_type_code": field_type_code,
                "field_type_name": FIELD_TYPE_MAPPING.get(field_type_code, f"未知类型({field_type_code})"),
                "is_primary": field.get("is_primary", False)
            }
            if field_type_code in [3, 4]:
                property_data = field.get("property", {})
                if property_data and "options" in property_data:
                    options = []
                    for opt in property_data["options"]:
                        options.append(opt.get("name"))
                    field_info["options"] = options
            fields_list.append(field_info)

        tables_info.append({
            "table_id": table_id,
            "table_name": table_name,
            "record_count": len(records),
            "view_count": len(views),
            "fields": fields_list
        })
    
    try:
        permissions = get_advanced_permissions(base_token, headers)
    except Exception as e:
        permissions = {"roles": [], "error": str(e)}

    return {
        "base_token": base_token,
        "tables": tables_info,
        "permissions": permissions
    }


def parse_bitable_url(feishu_url):
    try:
        table_info = extract_bitable_info_with_permissions(feishu_url)
        return json.dumps(table_info, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"解析失败: {str(e)}"}, ensure_ascii=False)
