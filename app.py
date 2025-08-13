"""
訪談智慧分析系統 - 企業級專業版
Professional Interview Analysis System with Modern UI
Author: Premium Version
License: Apache 2.0
"""

import os
import json
import time
import gradio as gr
import spaces
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from huggingface_hub import InferenceClient
import openai
from openai import OpenAI
import logging
import traceback
from functools import lru_cache
import hashlib
import re
import io
import base64
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# 測試用備用 API（如果沒有設置 OpenAI）
BACKUP_API_ENDPOINT = "https://api.openai.com/v1"  # 可替換為其他端點

HF_USERNAME = "s880453"
DATASET_NAME = "interview-transcripts-vectorized"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

INTERVIEWERS = ["徐美苓", "許弘諺", "郭禹彤"]

# Global variables
openai_client = None
hf_client = None
dataset = None
embedding_model = None
tokenizer = None
embedding_cache = {}

# ==========================================
# Data Models
# ==========================================

@dataclass
class SearchResult:
    content: str
    speaker: str
    turn_index: int
    file_id: str
    score: float
    chunk_index: int = 0
    context_type: str = "primary"
    weighted_score: float = 0.0
    reasoning: str = ""

# ==========================================
# Core System Functions
# ==========================================

def test_openai_connection():
    """測試 OpenAI API 連接"""
    try:
        if not OPENAI_API_KEY:
            return False, "未設置 OpenAI API Key"
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 測試調用
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        
        return True, "OpenAI API 連接成功"
        
    except Exception as e:
        logger.error(f"OpenAI connection test failed: {str(e)}")
        return False, f"OpenAI API 錯誤: {str(e)}"

def initialize_services():
    """初始化所有服務"""
    global openai_client, hf_client, dataset, embedding_model, tokenizer
    
    results = []
    
    try:
        logger.info("開始初始化服務...")
        
        # 測試並初始化 OpenAI
        if OPENAI_API_KEY:
            success, message = test_openai_connection()
            if success:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                results.append(("OpenAI", "✅", message))
            else:
                results.append(("OpenAI", "❌", message))
        else:
            results.append(("OpenAI", "⚠️", "未設置 API Key"))
        
        # 初始化 Hugging Face
        if HF_TOKEN:
            hf_client = InferenceClient(token=HF_TOKEN)
            results.append(("Hugging Face", "✅", "已連接"))
        else:
            results.append(("Hugging Face", "⚠️", "未設置 Token"))
        
        # 載入資料集
        try:
            dataset_path = f"{HF_USERNAME}/{DATASET_NAME}"
            dataset = load_dataset(dataset_path, token=HF_TOKEN, split="train")
            results.append(("資料集", "✅", f"已載入 {len(dataset)} 筆記錄"))
        except Exception as e:
            results.append(("資料集", "❌", str(e)))
            dataset = []  # 使用空資料集避免崩潰
        
        # 載入嵌入模型
        try:
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
            embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
            embedding_model.eval()
            
            if torch.cuda.is_available():
                embedding_model = embedding_model.cuda()
                results.append(("嵌入模型", "✅", "已載入 (GPU)"))
            else:
                results.append(("嵌入模型", "✅", "已載入 (CPU)"))
        except Exception as e:
            results.append(("嵌入模型", "❌", str(e)))
        
        return results
        
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        return [("系統", "❌", str(e))]

def get_speaker_list():
    """獲取所有受訪者列表"""
    try:
        if dataset and len(dataset) > 0:
            speakers = list(set([item.get('speaker', '') for item in dataset]))
            speakers = [s for s in speakers if s and s not in INTERVIEWERS]
            return sorted(speakers)
        return []
    except Exception as e:
        logger.error(f"Error getting speaker list: {str(e)}")
        return []

def average_pool(last_hidden_states, attention_mask):
    """Average pooling for embeddings"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@spaces.GPU(duration=30)
def generate_embedding(text: str):
    """生成文本嵌入向量"""
    global embedding_cache
    
    if not embedding_model or not tokenizer:
        return None
    
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    
    try:
        query_text = f"query: {text}"
        inputs = tokenizer(
            [query_text],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            if not next(embedding_model.parameters()).is_cuda:
                embedding_model.cuda()
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        embedding_vector = embeddings.cpu().numpy()[0]
        embedding_cache[text_hash] = embedding_vector
        
        return embedding_vector
        
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        return None

def call_gpt_with_fallback(prompt: str, temperature: float = 0.3):
    """調用 GPT 或使用備用方案"""
    
    # 首先嘗試 OpenAI
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一個專業的訪談分析助手。請用繁體中文回答。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
    
    # 如果 OpenAI 失敗，嘗試 Hugging Face
    if hf_client:
        try:
            response = hf_client.text_generation(
                prompt,
                max_new_tokens=1000,
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"HF API error: {str(e)}")
    
    # 如果都失敗，返回基礎回應
    return f"無法生成 AI 回應。請確認 API 設置。\n\n原始查詢：{prompt[:100]}..."

def semantic_search(query: str, selected_speakers: Union[List[str], str], top_k: int = 30):
    """語義搜尋"""
    try:
        if not query or not dataset or len(dataset) == 0:
            return []
        
        query_embedding = generate_embedding(query)
        if query_embedding is None:
            return []
        
        results = []
        
        for idx, item in enumerate(dataset):
            speaker = item.get('speaker', '')
            
            if speaker in INTERVIEWERS:
                continue
            
            if selected_speakers and selected_speakers != 'all':
                if isinstance(selected_speakers, list):
                    if speaker not in selected_speakers:
                        continue
                elif isinstance(selected_speakers, str):
                    if speaker != selected_speakers:
                        continue
            
            item_embedding = np.array(item.get('embedding', []))
            if len(item_embedding) == 0:
                continue
                
            similarity = np.dot(query_embedding, item_embedding)
            
            results.append(SearchResult(
                content=item.get('text', ''),
                speaker=speaker,
                turn_index=item.get('turn_index', 0),
                file_id=item.get('file_id', ''),
                score=float(similarity),
                chunk_index=item.get('chunk_index', 0)
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}")
        return []

def intelligent_search_with_rerank(query: str, selected_speakers, top_k: int = 20):
    """智慧搜尋與重排序"""
    try:
        # 初步向量檢索
        results = semantic_search(query, selected_speakers, top_k * 2)
        
        if not results:
            return []
        
        # 如果有 GPT，進行重排序
        if openai_client:
            try:
                rerank_prompt = f"""
                請根據相關性對以下結果重新排序。
                
                查詢: {query}
                
                結果列表:
                """
                
                for i, result in enumerate(results[:20], 1):
                    rerank_prompt += f"""
                {i}. 發言人: {result.speaker}
                   內容: {result.content[:150]}...
                   Turn: {result.turn_index}
                """
                
                rerank_prompt += "\n請返回最相關的前10個結果編號，用逗號分隔"
                
                response = call_gpt_with_fallback(rerank_prompt)
                
                if response and ',' in response:
                    numbers = re.findall(r'\d+', response)
                    reranked = []
                    
                    for num_str in numbers:
                        idx = int(num_str) - 1
                        if 0 <= idx < len(results):
                            results[idx].weighted_score = 0.7 + 0.3 * results[idx].score
                            reranked.append(results[idx])
                    
                    return reranked[:top_k] if reranked else results[:top_k]
            except:
                pass
        
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Reranking error: {str(e)}")
        return []

def process_word_document(file):
    """處理 Word 文檔"""
    try:
        import docx
        doc = docx.Document(file.name)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except ImportError:
        return "需要安裝 python-docx 套件來處理 Word 檔案"
    except Exception as e:
        return f"處理 Word 檔案錯誤: {str(e)}"

# ==========================================
# 現代化 Gradio 介面
# ==========================================

def create_modern_app():
    """創建現代化的 ChatGPT 風格介面"""
    
    # 現代化的 CSS 設計
    modern_css = """
    /* 隱藏 Gradio 底部 */
    footer { display: none !important; }
    
    /* 全域樣式 - 仿 ChatGPT */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
        min-height: 100vh;
        margin: 0 !important;
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* 深色主題變數 */
    :root {
        --primary-color: #10a37f;
        --primary-hover: #0d8c6c;
        --bg-primary: #0f0f0f;
        --bg-secondary: #1a1a1a;
        --bg-chat: #212121;
        --text-primary: #ececec;
        --text-secondary: #a0a0a0;
        --border-color: #2d2d2d;
        --message-user: #2b2b2b;
        --message-bot: #1e1e1e;
    }
    
    /* 主容器布局 */
    .main-container {
        display: flex;
        height: 100vh;
        overflow: hidden;
    }
    
    /* 側邊欄樣式 */
    .sidebar {
        width: 260px;
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
        display: flex;
        flex-direction: column;
        transition: width 0.3s ease;
    }
    
    .sidebar.collapsed {
        width: 0;
        overflow: hidden;
    }
    
    /* 新對話按鈕 */
    .new-chat-btn {
        margin: 16px;
        padding: 12px;
        background: transparent;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        cursor: pointer;
        transition: all 0.3s;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
    }
    
    .new-chat-btn:hover {
        background: var(--bg-chat);
        border-color: var(--primary-color);
    }
    
    /* 功能選項卡 */
    .sidebar-tabs {
        flex: 1;
        overflow-y: auto;
        padding: 0 16px;
    }
    
    .sidebar-tab {
        padding: 12px;
        margin: 8px 0;
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .sidebar-tab:hover {
        background: var(--bg-chat);
        color: var(--text-primary);
    }
    
    .sidebar-tab.active {
        background: var(--bg-chat);
        color: var(--primary-color);
    }
    
    /* 主聊天區域 */
    .chat-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: var(--bg-primary);
        position: relative;
    }
    
    /* 頂部工具欄 */
    .chat-header {
        height: 60px;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        padding: 0 24px;
        background: var(--bg-secondary);
    }
    
    .chat-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* 對話區域 */
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px 0;
        display: flex;
        flex-direction: column;
    }
    
    /* 訊息樣式 */
    .message-wrapper {
        display: flex;
        padding: 20px 24px;
        gap: 16px;
        transition: background 0.2s;
    }
    
    .message-wrapper:hover {
        background: rgba(255, 255, 255, 0.02);
    }
    
    .message-wrapper.user {
        background: var(--message-user);
    }
    
    .message-wrapper.assistant {
        background: var(--message-bot);
    }
    
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        flex-shrink: 0;
    }
    
    .user .message-avatar {
        background: var(--primary-color);
        color: white;
    }
    
    .assistant .message-avatar {
        background: var(--bg-chat);
        color: var(--primary-color);
    }
    
    .message-content {
        flex: 1;
        color: var(--text-primary);
        line-height: 1.6;
        font-size: 15px;
        max-width: 800px;
    }
    
    /* 輸入區域 */
    .input-container {
        border-top: 1px solid var(--border-color);
        padding: 20px;
        background: var(--bg-secondary);
    }
    
    .input-wrapper {
        max-width: 800px;
        margin: 0 auto;
        position: relative;
    }
    
    .chat-input {
        width: 100%;
        background: var(--bg-chat);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 12px 50px 12px 16px;
        color: var(--text-primary);
        font-size: 15px;
        resize: none;
        outline: none;
        transition: border-color 0.2s;
    }
    
    .chat-input:focus {
        border-color: var(--primary-color);
    }
    
    .send-button {
        position: absolute;
        right: 8px;
        bottom: 8px;
        width: 36px;
        height: 36px;
        background: var(--primary-color);
        border: none;
        border-radius: 8px;
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }
    
    .send-button:hover {
        background: var(--primary-hover);
        transform: scale(1.05);
    }
    
    .send-button:disabled {
        background: var(--border-color);
        cursor: not-allowed;
    }
    
    /* 文件上傳區域 */
    .upload-area {
        margin: 16px;
        padding: 20px;
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: var(--primary-color);
        background: rgba(16, 163, 127, 0.05);
    }
    
    .upload-area.dragging {
        border-color: var(--primary-color);
        background: rgba(16, 163, 127, 0.1);
    }
    
    /* 標籤樣式 */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 8px;
    }
    
    .badge.success {
        background: rgba(16, 163, 127, 0.2);
        color: var(--primary-color);
    }
    
    .badge.warning {
        background: rgba(255, 195, 0, 0.2);
        color: #ffc300;
    }
    
    .badge.error {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    /* 狀態指示器 */
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: var(--bg-chat);
        border-radius: 8px;
        margin: 16px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-dot.online {
        background: var(--primary-color);
    }
    
    .status-dot.offline {
        background: #ef4444;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* 卡片樣式 */
    .info-card {
        background: var(--bg-chat);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        margin: 16px;
    }
    
    .info-card h3 {
        color: var(--text-primary);
        margin: 0 0 12px 0;
        font-size: 16px;
    }
    
    .info-card p {
        color: var(--text-secondary);
        margin: 8px 0;
        font-size: 14px;
    }
    
    /* 快速操作按鈕 */
    .quick-action {
        display: inline-block;
        padding: 8px 16px;
        background: var(--bg-chat);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        color: var(--text-primary);
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
        margin: 4px;
    }
    
    .quick-action:hover {
        border-color: var(--primary-color);
        color: var(--primary-color);
        transform: translateY(-2px);
    }
    
    /* 響應式設計 */
    @media (max-width: 768px) {
        .sidebar {
            position: absolute;
            z-index: 1000;
            height: 100%;
        }
        
        .chat-container {
            width: 100%;
        }
        
        .message-content {
            font-size: 14px;
        }
    }
    
    /* 動畫效果 */
    .fade-in {
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 加載動畫 */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 20px 24px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: var(--text-secondary);
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    /* Gradio 覆蓋樣式 */
    .gr-button-primary {
        background: var(--primary-color) !important;
        border: none !important;
        color: white !important;
    }
    
    .gr-button-primary:hover {
        background: var(--primary-hover) !important;
    }
    
    .gr-input, .gr-textarea {
        background: var(--bg-chat) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .gr-input:focus, .gr-textarea:focus {
        border-color: var(--primary-color) !important;
    }
    
    .gr-box {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* 滾動條美化 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    """
    
    with gr.Blocks(
        title="訪談智慧分析系統",
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="gray",
            neutral_hue="gray",
            font=("Inter", "sans-serif")
        ),
        css=modern_css
    ) as app:
        
        # 狀態管理
        conversation_history = gr.State([])
        current_speakers = gr.State([])
        system_status = gr.State({})
        
        # 初始化系統並獲取狀態
        init_results = initialize_services()
        speakers = get_speaker_list()
        
        # 主容器
        with gr.Row(elem_classes="main-container"):
            
            # 左側邊欄
            with gr.Column(scale=1, elem_classes="sidebar"):
                # Logo 和標題
                gr.HTML("""
                    <div style="padding: 20px; border-bottom: 1px solid var(--border-color);">
                        <h2 style="color: var(--primary-color); margin: 0; font-size: 24px; text-align: center;">
                            🎙️ Interview AI
                        </h2>
                        <p style="color: var(--text-secondary); margin: 5px 0 0 0; font-size: 12px; text-align: center;">
                            Professional Analysis System
                        </p>
                    </div>
                """)
                
                # 系統狀態
                gr.HTML(f"""
                    <div class="status-indicator">
                        <div class="status-dot {'online' if any('✅' in str(r) for r in init_results) else 'offline'}"></div>
                        <span style="color: var(--text-secondary); font-size: 14px;">
                            {'系統運行中' if any('✅' in str(r) for r in init_results) else '部分功能離線'}
                        </span>
                    </div>
                """)
                
                # 功能標籤
                with gr.Tabs(elem_classes="sidebar-tabs"):
                    with gr.Tab("💬 對話設定"):
                        gr.Markdown("### 發言人篩選", elem_classes="info-card")
                        
                        speaker_mode = gr.Radio(
                            choices=["全部受訪者", "指定受訪者"],
                            value="全部受訪者",
                            label="",
                            elem_classes="radio-group"
                        )
                        
                        selected_speakers = gr.CheckboxGroup(
                            choices=speakers,
                            label="選擇受訪者",
                            value=[],
                            visible=False
                        )
                        
                        use_ai_rerank = gr.Checkbox(
                            label="🚀 AI 智慧排序",
                            value=True,
                            info="使用 GPT-4 提升準確度"
                        )
                        
                        search_depth = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=15,
                            step=5,
                            label="搜尋深度",
                            info="影響回答的詳細程度"
                        )
                    
                    with gr.Tab("📄 文件處理"):
                        gr.Markdown("### 上傳訪綱文件", elem_classes="info-card")
                        
                        word_upload = gr.File(
                            label="上傳 Word 檔案",
                            file_types=[".docx", ".doc"],
                            elem_classes="upload-area"
                        )
                        
                        guide_text = gr.Textbox(
                            label="或直接輸入訪綱",
                            placeholder="請輸入訪綱問題，每行一個...",
                            lines=5
                        )
                        
                        process_guide_btn = gr.Button(
                            "📋 分析訪綱",
                            variant="primary",
                            elem_classes="gr-button-primary"
                        )
                    
                    with gr.Tab("🔍 進階功能"):
                        gr.Markdown("### 上下文檢索", elem_classes="info-card")
                        
                        turn_index = gr.Number(
                            label="Turn Index",
                            value=50,
                            precision=0
                        )
                        
                        context_window = gr.Slider(
                            minimum=5,
                            maximum=20,
                            value=10,
                            label="上下文範圍"
                        )
                        
                        get_context_btn = gr.Button(
                            "📖 獲取上下文",
                            variant="primary"
                        )
                    
                    with gr.Tab("ℹ️ 系統資訊"):
                        # 顯示系統狀態
                        status_html = "<div class='info-card'><h3>系統狀態</h3>"
                        for service, status, message in init_results:
                            badge_class = "success" if "✅" in status else "error" if "❌" in status else "warning"
                            status_html += f"""
                            <p>
                                <span class='badge {badge_class}'>{status}</span>
                                <strong>{service}:</strong> {message}
                            </p>
                            """
                        status_html += "</div>"
                        
                        gr.HTML(status_html)
                        
                        gr.Markdown(f"""
                        <div class='info-card'>
                            <h3>資料統計</h3>
                            <p>📊 總記錄數: {len(dataset) if dataset else 0}</p>
                            <p>👥 受訪者數: {len(speakers)}</p>
                            <p>🎙️ 採訪者: {', '.join(INTERVIEWERS)}</p>
                            <p>📝 Turn 範圍: 0-317</p>
                        </div>
                        
                        <div class='info-card'>
                            <h3>技術規格</h3>
                            <p>🧠 語言模型: GPT-4o mini</p>
                            <p>🔍 向量模型: E5-Large</p>
                            <p>⚡ 加速: ZeroGPU</p>
                            <p>📦 版本: 2.0 Premium</p>
                        </div>
                        """)
            
            # 右側主對話區
            with gr.Column(scale=3, elem_classes="chat-container"):
                # 頂部標題欄
                gr.HTML("""
                    <div class="chat-header">
                        <div class="chat-title">
                            <span>🤖</span>
                            <span>AI 訪談分析助手</span>
                        </div>
                    </div>
                """)
                
                # 對話區域
                chatbot = gr.Chatbot(
                    label="",
                    elem_id="chatbot",
                    height=500,
                    show_label=False,
                    avatar_images=["👤", "🤖"]
                )
                
                # 快速操作建議
                gr.HTML("""
                    <div style="padding: 16px; text-align: center;">
                        <span style="color: var(--text-secondary); margin-right: 12px;">快速提問：</span>
                        <span class="quick-action" onclick="document.querySelector('textarea').value='受訪者對環保議題的看法？'">
                            環保議題
                        </span>
                        <span class="quick-action" onclick="document.querySelector('textarea').value='比較不同受訪者的觀點差異'">
                            觀點比較
                        </span>
                        <span class="quick-action" onclick="document.querySelector('textarea').value='找出關於永續發展的討論'">
                            永續發展
                        </span>
                        <span class="quick-action" onclick="document.querySelector('textarea').value='誰提到了政策建議？'">
                            政策建議
                        </span>
                    </div>
                """)
                
                # 輸入區域
                with gr.Row(elem_classes="input-container"):
                    msg = gr.Textbox(
                        label="",
                        placeholder="輸入訊息... (Shift+Enter 換行)",
                        lines=2,
                        max_lines=5,
                        show_label=False,
                        elem_classes="chat-input",
                        scale=9
                    )
                    
                    send_btn = gr.Button(
                        "➤",
                        elem_classes="send-button",
                        scale=1
                    )
                
                # 底部提示
                gr.HTML("""
                    <div style="text-align: center; padding: 10px; color: var(--text-secondary); font-size: 12px;">
                        💡 提示：可以詢問特定受訪者的觀點、比較不同意見、或探索特定主題
                    </div>
                """)
        
        # ==========================================
        # 事件處理函數
        # ==========================================
        
        def process_message(message, history, speakers_filter, use_rerank, depth):
            """處理對話訊息"""
            if not message:
                return "", history
            
            # 顯示用戶訊息
            history = history or []
            
            try:
                # 檢查系統狀態
                if not dataset or len(dataset) == 0:
                    response = "⚠️ 資料集未載入。請檢查系統設置。"
                    history.append([message, response])
                    return "", history
                
                # 決定搜尋範圍
                search_speakers = speakers_filter if speakers_filter else "all"
                
                # 執行智慧搜尋
                if use_rerank and openai_client:
                    results = intelligent_search_with_rerank(message, search_speakers, depth)
                else:
                    results = semantic_search(message, search_speakers, depth)
                
                if results:
                    # 準備上下文
                    context = "\n\n".join([
                        f"【{r.speaker}】Turn {r.turn_index}:\n{r.content[:250]}..."
                        for r in results[:5]
                    ])
                    
                    # 生成回答
                    prompt = f"""
                    基於以下訪談內容，請提供專業且詳細的回答。
                    
                    用戶問題：{message}
                    
                    相關訪談內容：
                    {context}
                    
                    要求：
                    1. 直接且準確地回答問題
                    2. 引用具體的發言人觀點
                    3. 如有不同觀點請對比說明
                    4. 保持專業但友善的語氣
                    
                    請用繁體中文回答。
                    """
                    
                    response = call_gpt_with_fallback(prompt)
                    
                    # 添加來源引用
                    if len(results) > 0:
                        response += "\n\n---\n**📚 資料來源：**\n"
                        for i, r in enumerate(results[:3], 1):
                            response += f"{i}. {r.speaker} (Turn {r.turn_index}) - 相關度: {r.score:.2%}\n"
                else:
                    response = "😔 抱歉，我在訪談記錄中找不到相關內容。請嘗試：\n1. 使用不同的關鍵詞\n2. 擴大搜尋範圍\n3. 確認問題是否與訪談主題相關"
                
                history.append([message, response])
                
            except Exception as e:
                logger.error(f"處理訊息錯誤: {str(e)}")
                response = f"❌ 處理錯誤：{str(e)}\n\n請檢查系統設置或聯繫管理員。"
                history.append([message, response])
            
            return "", history
        
        def process_word_file(file):
            """處理上傳的 Word 檔案"""
            if not file:
                return ""
            
            try:
                text = process_word_document(file)
                return text
            except Exception as e:
                return f"處理檔案錯誤: {str(e)}"
        
        def analyze_guide(text, speakers_filter):
            """分析訪綱"""
            if not text:
                return "請提供訪綱內容"
            
            questions = [q.strip() for q in text.split('\n') if q.strip()]
            
            result = "# 📋 訪綱分析結果\n\n"
            
            for i, question in enumerate(questions, 1):
                result += f"## 問題 {i}: {question}\n\n"
                
                # 搜尋相關內容
                results = intelligent_search_with_rerank(
                    question,
                    speakers_filter if speakers_filter else "all",
                    10
                )
                
                if results:
                    for j, r in enumerate(results[:3], 1):
                        result += f"### 回答 {j}\n"
                        result += f"**發言人:** {r.speaker} (Turn {r.turn_index})\n"
                        result += f"**內容:** {r.content[:300]}...\n"
                        result += f"**相關度:** {r.score:.2%}\n\n"
                else:
                    result += "*未找到相關回答*\n\n"
                
                result += "---\n\n"
            
            return result
        
        def get_context(turn_idx, window):
            """獲取上下文"""
            try:
                turn_idx = int(turn_idx)
                window = int(window)
                
                start = max(0, turn_idx - window)
                end = turn_idx + window
                
                result = f"# 📖 Turn {turn_idx} 上下文 (±{window})\n\n"
                
                context_items = []
                for item in dataset:
                    item_turn = item.get('turn_index', 0)
                    if start <= item_turn <= end:
                        context_items.append(item)
                
                context_items.sort(key=lambda x: x.get('turn_index', 0))
                
                for item in context_items:
                    item_turn = item.get('turn_index', 0)
                    speaker = item.get('speaker', '')
                    content = item.get('text', '')
                    
                    if item_turn == turn_idx:
                        result += f"## 🎯 **Turn {item_turn}** - {speaker}\n"
                    else:
                        result += f"## Turn {item_turn} - {speaker}\n"
                    
                    result += f"> {content[:500]}...\n\n"
                
                return result
                
            except Exception as e:
                return f"錯誤: {str(e)}"
        
        def toggle_speakers(mode):
            """切換發言人選擇"""
            return gr.update(visible=(mode == "指定受訪者"))
        
        # ==========================================
        # 事件綁定
        # ==========================================
        
        # 主對話功能
        send_btn.click(
            fn=process_message,
            inputs=[msg, chatbot, selected_speakers, use_ai_rerank, search_depth],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            fn=process_message,
            inputs=[msg, chatbot, selected_speakers, use_ai_rerank, search_depth],
            outputs=[msg, chatbot]
        )
        
        # 發言人模式切換
        speaker_mode.change(
            fn=toggle_speakers,
            inputs=[speaker_mode],
            outputs=[selected_speakers]
        )
        
        # Word 檔案處理
        word_upload.change(
            fn=process_word_file,
            inputs=[word_upload],
            outputs=[guide_text]
        )
        
        # 訪綱分析
        process_guide_btn.click(
            fn=analyze_guide,
            inputs=[guide_text, selected_speakers],
            outputs=[chatbot]
        )
        
        # 上下文檢索
        get_context_btn.click(
            fn=get_context,
            inputs=[turn_index, context_window],
            outputs=[chatbot]
        )
        
        return app

# ==========================================
# 主程式入口
# ==========================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 啟動訪談智慧分析系統 - 企業級專業版")
    logger.info("=" * 60)
    
    # 創建應用
    try:
        app = create_modern_app()
        
        # 啟動應用
        app.queue(max_size=20)
        app.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"❌ 應用啟動失敗: {str(e)}")
        
        # 顯示錯誤介面
        with gr.Blocks(title="系統錯誤", css="body { background: #1a1a1a; color: white; }") as error_app:
            gr.HTML("""
                <div style="max-width: 800px; margin: 50px auto; padding: 40px; 
                           background: #2d2d2d; border-radius: 12px; text-align: center;">
                    <h1 style="color: #ef4444;">❌ 系統啟動失敗</h1>
                    <p style="color: #a0a0a0; margin: 20px 0;">請檢查以下設置：</p>
                    <div style="text-align: left; background: #1a1a1a; padding: 20px; 
                               border-radius: 8px; margin: 20px 0;">
                        <p>1. <strong>OPENAI_API_KEY</strong> - 在 Settings → Repository secrets 設置</p>
                        <p>2. <strong>HF_TOKEN</strong> - 在 Settings → Repository secrets 設置</p>
                        <p>3. 確認資料集 <strong>s880453/interview-transcripts-vectorized</strong> 可訪問</p>
                        <p>4. 檢查 GPU 配額是否充足</p>
                    </div>
                    <p style="color: #ffc300;">💡 設置完成後請重新啟動 Space</p>
                </div>
            """)
        
        error_app.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860
        )
