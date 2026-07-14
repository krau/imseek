# imseek

[English](README.md) | [中文](README.zh-CN.md)

纯 Go 实现的基于 ORB 特征点匹配的以图搜图.

## 后端

| 后端       | 存储                      | 索引            |
| ---------- | ------------------------- | --------------- |
| `local`    | SQLite + 本地文件         | IVF             |
| `pgvector` | PostgreSQL（元数据+向量） | IVFFlat 或 HNSW |
| `milvus`   | Milvus                    | BIN_IVF_FLAT    |

### local

`local` 是默认的后端, 无需任何外部服务与构建依赖步骤, 开箱即用.

`local` 后端的实现几乎完全参考自 [imsearch](https://github.com/lolishinshi/imsearch), 可以认为是对其的 Go 重写. 感谢前人的智慧.

数据目录:

```
data/
├── imseek.db          # 元数据、描述子、索引信息
├── quantizer.bin      # 粗量化器
└── invlists.bin       # 倒排列表
```

### pgvector

元数据和向量全部存储于 PostgreSQL. 需安装 pgvector 插件. 支持 IVFFlat 和 HNSW 两种索引方式.

IVFFlat 的构建速度更快, 但召回率差, HNSW 则反之.

## 构建

```bash
# 默认（仅 local 后端）
go build -o imseek ./cmd/imseek

# 包含 pgvector 支持
go build -tags pgvector -o imseek ./cmd/imseek

# 包含 milvus 支持
go build -tags milvus -o imseek ./cmd/imseek

# 全部后端
go build -tags 'pgvector milvus' -o imseek ./cmd/imseek
```

## 使用

### local 后端

```bash
# 1. 添加图片
./imseek add /path/to/images

# 2. 训练量化器
./imseek train

# 3. 构建索引
./imseek build

# 4. 搜索
./imseek search query.jpg

# 5. 启动 HTTP 服务
./imseek server --token my-secret
```

### pgvector 后端

需要为 postgresql 安装 pgvector 插件, 或使用 Docker 快速启动一个具有 pgvector 的实例:

```bash
docker run -d --name imseek-pg --shm-size=1g \
  -e POSTGRES_USER=imseek -e POSTGRES_PASSWORD=imseek \
  -e POSTGRES_DB=imseek -p 5432:5432 pgvector/pgvector:pg18
```

添加图片

```bash
./imseek --backend pgvector \
  --pgvector.conn_string 'postgres://imseek:imseek@localhost:5432/imseek?sslmode=disable' \
  add /path/to/images
```

搜索

```bash
./imseek --backend pgvector \
  --pgvector.conn_string 'postgres://imseek:imseek@localhost:5432/imseek?sslmode=disable' \
  search query.jpg
```

pgvector 后端无需手动执行训练和构建索引, 因为 pgvector 会在添加图片时自动完成这些操作.

## HTTP API

```bash
# 启动服务
./imseek server --token my-secret

# 添加图片
curl -H "Authorization: Bearer my-secret" \
  -F "file=@photo.jpg" http://localhost:8000/api/v1/images

# 搜索
curl -H "Authorization: Bearer my-secret" \
  -F "file=@query.jpg" http://localhost:8000/api/v1/search

# 删除
curl -X DELETE -H "Authorization: Bearer my-secret" \
  http://localhost:8000/api/v1/images/1

# 统计
curl -H "Authorization: Bearer my-secret" \
  http://localhost:8000/api/v1/stats
```

Swagger 文档：`http://localhost:8000/swagger`

## 配置

使用配置文件（`imseek.toml`）、环境变量（`IMSEEK_*`）或命令行参数。详见 `imseek.example.toml`。

主要搜索参数：

| 参数                 | 默认值 | 说明                           |
| -------------------- | ------ | ------------------------------ |
| `search.distance`    | 64     | 最大 Hamming 距离              |
| `search.count`       | 10     | 返回结果数量                   |
| `search.k`           | 3      | 每个描述子的 KNN 数量          |
| `search.nprobe`      | 3      | 探测的倒排列表数（local）      |
| `pgvector.probes`    | 10     | 探测的桶数（pgvector IVFFlat） |
| `pgvector.ef_search` | 40     | 候选列表大小（pgvector HNSW）  |

## 致谢

- [imsearch](https://github.com/lolishinshi/imsearch)
- [ORB_SLAM3](https://github.com/raulmur/ORB_SLAM3)