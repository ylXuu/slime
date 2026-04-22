import requests
import random
import time
import json
import asyncio
from pprint import pprint
import os
import sys
from typing import Dict, Any, List
from pathlib import Path

import aiohttp
import asyncio
from typing import Dict, List, Any




class AsyncSearchBrowserClient:
    def __init__(self, address, port, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None
        self.server_addr = f"http://{address}:{port}"
        # print(self.server_list)     
        
    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        cnt = 0
        last_exception = None
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_addr}/retrieve",
                        json=req_meta,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120)
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                documents=[r["contents"] for r in result],
                                urls=[r["url"] for r in result],
                                server_type="async-search-browser",
                            ) for result in res["result"]
                        ]
            except Exception as e:
                print("query_async", e.__class__.__name__, e.__cause__)
                last_exception = e
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)
        
        raise RuntimeError("Fail to post search query to RAG server") from last_exception
        
    async def access_async(self, urls: List[str]) -> List[Dict]:
        cnt = 0
        last_exception = None        
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_addr}/access",
                        json={"urls": urls},
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120)
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                page=result["contents"] if result is not None else "",
                                type="access",
                                server_type="async-search-browser",
                            ) for result in res["result"]
                        ]
            except Exception as e:
                print("access_async", e.__class__.__name__, e.__cause__)
                last_exception = e
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)
        
        raise RuntimeError("Fail to post access request to RAG server") from last_exception

