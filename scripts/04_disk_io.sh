#!/bin/bash
# 脚本：04_disk_io.sh
# 用途：测试磁盘顺序读写速度，对应报告第 3.2 节
# 说明：测试 overlay 层和 NVMe 路径，4GB 文件，Direct IO

echo "========================================"
echo "  磁盘 I/O 性能测试 (Direct IO, 4GB)"
echo "========================================"

TEST_FILE_SIZE="4G"
BLOCK_SIZE="1G"

test_path() {
    local path=$1
    local label=$2

    if [ ! -d "$path" ]; then
        echo "  [$label] 路径不存在，跳过: $path"
        return
    fi

    echo ""
    echo "  [$label] 路径: $path"

    # 顺序写测试
    echo -n "    顺序写 ($TEST_FILE_SIZE): "
    WRITE_RESULT=$(dd if=/dev/zero of="${path}/test_write.tmp" \
        bs=$BLOCK_SIZE count=4 oflag=direct conv=fdatasync 2>&1 | \
        grep -oP '\d+(\.\d+)? [GM]B/s' | tail -1)
    echo "${WRITE_RESULT:-测试失败}"

    sync

    # 顺序读测试（先清缓存）
    echo -n "    顺序读 ($TEST_FILE_SIZE): "
    # 清页缓存（需 root）
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    READ_RESULT=$(dd if="${path}/test_write.tmp" of=/dev/null \
        bs=$BLOCK_SIZE iflag=direct 2>&1 | \
        grep -oP '\d+(\.\d+)? [GM]B/s' | tail -1)
    echo "${READ_RESULT:-测试失败}"

    # 清理
    rm -f "${path}/test_write.tmp"
}

# 测试 overlay 文件系统（容器层）
test_path "/tmp" "overlay/容器层 (/tmp)"

# 测试 NVMe 路径（挂载点，按实际路径调整）
test_path "/anc-init" "NVMe 分区 (/anc-init)"

# 测试 NFS 共享（按实际挂载点调整）
test_path "/root/public" "NFS 共享 (/root/public)"

echo ""
echo "参考值："
echo "  overlay 写: ~1.1 GB/s | 读: ~1.9 GB/s（容器层受限）"
echo "  NVMe 写:    ~3-7 GB/s | 读: ~3-7 GB/s（裸 NVMe）"
echo "========================================"
