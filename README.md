# OneFormer 单图水域分割（仅 `swin_large`）

本项目仅保留一个模型：

- `shi-labs/oneformer_ade20k_swin_large`

输出：

- `output/water_mask.png`
- `output/water_overlay.png`
- `output/water_coords.json`
- `output/stats.json`

## 1. 环境准备

```bash
chmod +x scripts/setup_linux_cpu.sh
./scripts/setup_linux_cpu.sh
```

说明：

- 该脚本会在项目目录创建 `.venv`，并把依赖都安装到 `.venv` 内。
- 脚本会强制安装 CPU 版 PyTorch（Linux x86_64）。
- 后续命令建议统一使用项目环境：

```bash
source .venv/bin/activate
python -V
```

## 2. 下载权重（必须先执行）

```bash
python download_weights.py
```

说明：

- 推理阶段不会自动下载模型（只读取本地目录）。
- 权重目录固定为 `models/oneformer_ade20k_swin_large`。
- 若目录已完整，下载脚本会自动跳过，不重复下载。

## 3. 推理

```bash
python src/main.py --image input/test.jpg
```

可选参数：

- `--output-dir`：输出目录（默认 `output`）
- `--min-area`：连通域面积阈值（默认 `300`）
- `--alpha`：overlay 透明度（默认 `0.4`）

## 4. 单元测试

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 5. HTTP 接口

启动服务：

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

可选环境变量（Linux 部署建议）：

```bash
export OUTPUT_API_DIR=/data/oneformer/output_api
export OUTPUT_RETENTION_HOURS=24
```

说明：

- `OUTPUT_API_DIR`：接口输出持久化目录（默认 `output/api`）。
- `OUTPUT_RETENTION_HOURS`：保留时长（小时，默认 `24`）。
- 服务每次调用 `/segment` 时，都会自动清理该目录下 **24 小时前**（或你设置时长前）的历史任务目录/文件。

基础地址示例：`http://127.0.0.1:8000`

### 5.1 健康检查

- 方法：`GET /health`
- 响应：

```json
{
  "status": "ok"
}
```

### 5.2 分割接口

- 方法：`POST /segment`
- `Content-Type`：`application/json`
- 功能：输入图片（URL 或 base64），返回 mask/overlay 下载地址、归一化多边形坐标和数量。

请求体字段：

- `image_url`：`string`，图片 URL（与 `image_base64` 二选一）
- `image_base64`：`string`，图片 base64（支持 data URL 前缀，与 `image_url` 二选一）
- `min_area`：`int`，可选，默认 `300`，连通域最小面积过滤阈值
- `simplify_tolerance`：`float`，可选，默认 `2.0`，多边形简化容差
- `alpha`：`float`，可选，默认 `0.4`，overlay 透明度，范围 `[0,1]`

参数解释（重点）：

- `min_area`  
  含义：后处理中连通域最小保留面积（像素）。小于该值的水域小块会被过滤。  
  影响：值越大，噪点越少，但也更容易丢失小水域。  
  建议：  
  - 噪点很多时可提高到 `500~2000`  
  - 需要保留细小水体时可降到 `0~200`

- `simplify_tolerance`  
  含义：多边形轮廓简化容差（单位：像素，基于 `approxPolyDP`）。  
  影响：值越大，返回点数越少、轮廓越平滑；值越小，轮廓更贴边但点数更多。  
  建议：  
  - 精细边界：`0.5~1.5`  
  - 通用场景：`2.0`（默认）  
  - 需要更少点数：`3.0~8.0`

- `alpha`  
  含义：叠加图中水域颜色透明度。  
  影响：`0` 表示几乎不着色，`1` 表示完全用叠加色覆盖。  
  建议：  
  - 更接近原图：`0.2~0.35`  
  - 平衡可视化：`0.4`（默认）  
  - 强强调水域：`0.5~0.7`

约束：

- `image_url` 与 `image_base64` 必须且只能提供一个。

请求示例（URL）：

```json
{
  "image_url": "https://example.com/test.jpg",
  "min_area": 300,
  "simplify_tolerance": 2.0,
  "alpha": 0.4
}
```

请求示例（base64）：

```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "min_area": 300,
  "simplify_tolerance": 2.0,
  "alpha": 0.4
}
```

响应包含：

- `mask_url`（已存储 mask 图片 URL）
- `overlay_url`（已存储 overlay 图片 URL）
- `polygons_xy`（多个多边形归一化坐标，范围 `[0,1]`）
- `polygon_count`
- `polygons_file`（多边形 JSON 下载 URL）
- `task_id`（本次任务 ID）

响应示例：

```json
{
  "task_id": "d4c0b5b8441d45d5af5ac71da9159f6c",
  "mask_url": "http://127.0.0.1:8000/files/d4c0b5b8441d45d5af5ac71da9159f6c/water_mask.png",
  "overlay_url": "http://127.0.0.1:8000/files/d4c0b5b8441d45d5af5ac71da9159f6c/water_overlay.png",
  "polygons_xy": [
    [[0.12, 0.35], [0.18, 0.34], [0.19, 0.41], [0.13, 0.42]]
  ],
  "polygon_count": 1,
  "polygons_file": "http://127.0.0.1:8000/files/d4c0b5b8441d45d5af5ac71da9159f6c/water_polygons.json"
}
```

### 5.3 文件下载接口

- 方法：`GET /files/{file_path}`
- 说明：用于下载 `mask_url` / `overlay_url` / `polygons_file` 指向的文件。
- 示例：
  - `GET /files/<task_id>/water_mask.png`
  - `GET /files/<task_id>/water_overlay.png`
  - `GET /files/<task_id>/water_polygons.json`

### 5.4 cURL 调用示例

URL 输入：

```bash
curl -X POST "http://127.0.0.1:8000/segment" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/test.jpg","min_area":300,"simplify_tolerance":2.0,"alpha":0.4}'
```

base64 输入（示例）：

```bash
curl -X POST "http://127.0.0.1:8000/segment" \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"data:image/png;base64,XXXX...","min_area":300,"simplify_tolerance":2.0,"alpha":0.4}'
```

### 5.5 错误码说明

- `400`：输入非法（URL 下载失败、base64 非法、图片解码失败、参数不满足约束）
- `404`：请求的文件不存在（`/files/...`）
- `500`：服务内部错误（如输出文件写入失败）
