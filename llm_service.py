import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="News Summarization and Translation Service")

# Initialize Anthropic client
api_key = os.getenv("CLAUDE_API_KEY")
if not api_key:
    raise ValueError("CLAUDE_API_KEY environment variable not set")

client = AsyncAnthropic(api_key=api_key)

# Define request and response models
class ArticleRequest(BaseModel):
    title: str
    content: str
    url: Optional[str] = None

class ArticleResponse(BaseModel):
    summarized_title: str
    summarized_content: str
    original_title: str
    original_url: Optional[str] = None

class BatchArticleRequest(BaseModel):
    articles: List[ArticleRequest]

class BatchArticleResponse(BaseModel):
    results: List[ArticleResponse]
    failed_indices: List[int] = []

# System prompt for Claude
SYSTEM_PROMPT = """
You are a specialized AI that summarizes and translates news articles from English to Korean.

Your task is to:
1. Analyze the provided news article
2. Create a concise summary in Korean (maximum 200 characters)
3. Create a short, descriptive title in Korean (maximum 50 characters)

Follow these rules:
- The summary should capture the main points of the article
- The title should be descriptive and attention-grabbing
- All output must be in Korean
- Maintain factual accuracy
- Do not add information not present in the original article
- Format your response as a JSON object with keys 'summarized_title' and 'summarized_content'
"""

async def summarize_article(article: ArticleRequest) -> ArticleResponse:
    """Summarize and translate a single article using Claude API"""
    try:
        # Prepare the prompt
        user_prompt = f"""
Original Title: {article.title}
Original URL: {article.url if article.url else 'Not provided'}

Original Content:
{article.content}

Please summarize this article in Korean and provide a Korean title.
Return ONLY a valid JSON object with 'summarized_title' and 'summarized_content' keys.
"""

        # Call Claude API
        response = await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        # Extract and parse the response
        content = response.content[0].text
        
        # Try to extract JSON from the response
        try:
            # Find JSON object in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole response
                result = json.loads(content)
                
            return ArticleResponse(
                summarized_title=result.get('summarized_title', '제목 없음'),
                summarized_content=result.get('summarized_content', '내용 없음'),
                original_title=article.title,
                original_url=article.url
            )
        except json.JSONDecodeError:
            # If JSON parsing fails, extract content manually
            lines = content.strip().split('\n')
            title = ""
            content_text = ""
            
            for line in lines:
                if not title and ("제목" in line or "타이틀" in line or "title" in line.lower()):
                    title = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                elif not title and len(line.strip()) < 50:
                    # Assume first short line is the title
                    title = line.strip()
                else:
                    content_text += line.strip() + " "
            
            return ArticleResponse(
                summarized_title=title[:50] or '제목 없음',
                summarized_content=content_text[:200] or '내용 없음',
                original_title=article.title,
                original_url=article.url
            )
            
    except Exception as e:
        print(f"Error summarizing article: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to summarize article: {str(e)}")

@app.post("/summarize", response_model=ArticleResponse)
async def summarize_single_article(article: ArticleRequest):
    """API endpoint to summarize a single article"""
    return await summarize_article(article)

@app.post("/batch-summarize", response_model=BatchArticleResponse)
async def summarize_batch_articles(request: BatchArticleRequest):
    """API endpoint to summarize multiple articles in parallel"""
    results = []
    failed_indices = []
    
    # Process articles in parallel
    tasks = []
    for i, article in enumerate(request.articles):
        tasks.append((i, asyncio.create_task(summarize_article(article))))
    
    # Collect results
    for i, task in tasks:
        try:
            result = await task
            results.append(result)
        except Exception:
            failed_indices.append(i)
            # Add a placeholder for failed articles
            results.append(ArticleResponse(
                summarized_title="처리 실패",
                summarized_content="이 기사는 처리 중 오류가 발생했습니다.",
                original_title=request.articles[i].title,
                original_url=request.articles[i].url
            ))
    
    return BatchArticleResponse(results=results, failed_indices=failed_indices)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_service:app", host="0.0.0.0", port=8000, reload=True)
