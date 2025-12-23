from __future__ import annotations
import os
import threading
from typing import List, Tuple, Optional
import typing
import requests
from joblib import Parallel, delayed
from tqdm import tqdm
from retry import retry
import zipfile
import platform
import gzip
import shutil

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE"
    )
}

MB = 1024 ** 2


def split_ranges(total_size: int, part_size: int) -> List[Tuple[int, int]]:
    """返回 (start, end) 闭区间，适用于 Range: bytes=start-end"""
    if total_size <= 0:
        return []
    ranges = []
    cur = 0
    last = total_size - 1
    while cur <= last:
        end = min(cur + part_size - 1, last)
        ranges.append((cur, end))
        cur = end + 1
    return ranges


def get_file_info(url: str, session: requests.Session) -> Tuple[Optional[int], bool]:
    """返回 (file_size, accept_ranges)"""
    resp = session.head(url, headers=headers, allow_redirects=True, timeout=15)
    resp.raise_for_status()

    length = resp.headers.get("Content-Length")
    accept_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
    return (int(length) if length is not None else None), accept_ranges


def probe_range(url: str, session: requests.Session) -> bool:
    """做一次 Range 探测：Range:0-0 若返回 206 说明支持"""
    try:
        h = headers.copy()
        h["Range"] = "bytes=0-0"
        r = session.get(url, headers=h, stream=True, timeout=15)
        ok = (r.status_code == 206)
        r.close()
        return ok
    except Exception:
        return False


def download_single(url: str, file_name: str) -> None:
    """单线程流式下载（不支持 Range 或拿不到长度时）"""
    with requests.Session() as s:
        with s.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(file_name, "wb") as f, tqdm(desc=f"下载文件：{file_name}", unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))


def download_joblib(
    url: str,
    file_name: str,
    retry_times: int = 3,
    each_size: int = 16 * MB,
    n_jobs: int = -1,          # -1 = 使用所有可用核心的线程数（joblib 的语义）
    prefer: str = "threads",   # I/O 下载任务用 threads
) -> None:
    with requests.Session() as session:
        file_size, accept_ranges = get_file_info(url, session)

        if not file_size:
            print("无法获取 Content-Length，改为单线程下载（不使用 Range）。")
            return download_single(url, file_name)

        range_ok = accept_ranges or probe_range(url, session)
        if not range_ok:
            print("服务器不支持 Range（分段下载），改为单线程下载。")
            return download_single(url, file_name)

    # 分块
    each_size = max(1, min(each_size, file_size))
    parts = split_ranges(file_size, each_size)
    print(f"分块数：{len(parts)}")

    # 预分配文件大小，避免并发扩展文件
    with open(file_name, "wb") as f:
        f.truncate(file_size)

    file_lock = threading.Lock()
    bar_lock = threading.Lock()
    bar = tqdm(total=file_size, desc=f"下载文件：{file_name}", unit="B", unit_scale=True)

    @retry(tries=retry_times, delay=1, backoff=2)
    def fetch_and_write(start: int, end: int) -> int:
        """下载一个分块并写入正确位置，返回实际写入字节数"""
        h = headers.copy()
        h["Range"] = f"bytes={start}-{end}"

        # 每个任务独立 Session，更稳
        with requests.Session() as s:
            with s.get(url, headers=h, stream=True, timeout=30) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"Range 请求失败，期望 206，实际 {r.status_code}")

                offset = start
                written = 0
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue

                    # 写入需要锁：seek+write
                    with file_lock:
                        with open(file_name, "r+b") as f:
                            f.seek(offset)
                            f.write(chunk)

                    offset += len(chunk)
                    written += len(chunk)

                    # tqdm 更新也加锁
                    with bar_lock:
                        bar.update(len(chunk))

                # 可选：校验每块长度
                expected = end - start + 1
                if written != expected:
                    raise RuntimeError(f"分块写入长度不一致：{start}-{end} 写入 {written} / 期望 {expected}")

                return written

    try:
        Parallel(n_jobs=n_jobs, prefer=prefer, batch_size=1)(
            delayed(fetch_and_write)(s, e) for s, e in parts
        )
    finally:
        bar.close()

def main(extmodule:typing.Literal['admixture','iqtree3']):
    scriptpath = os.path.dirname(os.path.abspath(__file__))
    platform_name = platform.system()
    assert platform_name in ['Linux', 'Darwin'], 'Only Linux and macOS are supported.'
    arc = {'Linux': 'x86_64', 'Darwin': 'universal'}
    os.makedirs(f'{scriptpath}/../../ext/bin/', exist_ok=True)
    extbin = f'{scriptpath}/../../ext/bin'
    # Denpendency check
    ## iqtree
    if not os.path.isfile(f'{extbin}/{extmodule}'):
        print(f'Downloading and compiling {extmodule}...')
        url = f'http://maize.jxfu.top:23000/Jingxian/JanusXext/raw/main/package/{extmodule}-{platform_name}-{arc[platform_name]}.gz'
        download_joblib(url, f'{extbin}/{extmodule}.gz', n_jobs=-1)
        with gzip.open(f'{extbin}/{extmodule}.gz', "rb") as f_in, open(f'{extbin}/{extmodule}', "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.chmod(f'{extbin}/{extmodule}', 0o755)
        os.remove(f'{extbin}/{extmodule}.gz')
        print(f'{extmodule} downloaded and compiled successfully.')
    ## Main
    return f'{extbin}/{extmodule}'

if __name__ == "__main__":
    url = "https://github.com/iqtree/iqtree3/releases/download/v3.0.1/iqtree-3.0.1-macOS.zip"
    file_name = os.path.basename(url)
    download_joblib(url, file_name, each_size=16 * MB, n_jobs=-1)
    
    zip_file_path = file_name
    # 打开并解压 ZIP 文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('.')