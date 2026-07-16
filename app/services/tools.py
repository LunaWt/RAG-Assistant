import simpleeval
import trafilatura
import asyncio
from trafilatura.settings import use_config
from simpleeval import simple_eval, InvalidExpression, NameNotDefined, FunctionNotDefined
from ddgs import DDGS
from ddgs.exceptions import DDGSException


from app.services.llm_client import generate_summary
from app.services.vector_db import vector_db


new_config = use_config()

new_config.set('DEFAULT', 'DOWNLOAD_TIMEOUT', '5')
simpleeval.MAX_POWER = 1000

def search_knowledge_base(
    query: str, 
    filename: str | None = None,
    ) -> str:
    """Rag system search with cosine similarity retrieval"""
    
    try:
        if filename:
            return '\n\n'.join(vector_db.rag_search(query, filename))
        else:
            return '\n\n'.join(vector_db.rag_search(query))
    except Exception as e:
        return f'Knowledge base search failed: {e}. Try web_search or answer from your own knowledge.'


def calculator(expression: str):
    """Evaluate a simple arithmetic expression, e.g. "2 + 2" or "2 + 1 * 9 / 16".

    Only numeric literals and arithmetic operators (+ - * / ** %) are allowed:
    no variables, no function calls (sqrt, sin etc. are NOT available).
    Expression must be a string up to 100 characters; powers are limited.
    Result is rounded to 4 decimal places."""
    
    if not isinstance(expression, str):
        return 'Выражение должно быть строкой'
    if len(expression) > 100:
        return "Слишком большое выражение"
    try:
        return round(
                number=simple_eval(expr=expression),
                ndigits=4)
    except (NameNotDefined, FunctionNotDefined) as e:
        return f'Использование запрещенных переменных или функций {e}'
    except InvalidExpression as e:
        return f'Ошибка в выражении: {e}'
    except SyntaxError:
        return 'Синтаксическая ошибка в выражении'
    except Exception as e:
        return f'Ошибка{e}'


def _ddg_search(query: str) -> list[dict]:
    return list(DDGS().text(
        query,
        region='wt-wt',
        safesearch='off',
        timelimit='y',
        max_results=5,
    ))
def _fetch_and_extract(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url, config=new_config)
    if not downloaded:
        return None
    return trafilatura.extract(downloaded, include_comments=False)
async def web_search(query: str) -> tuple[str, list[dict]]:
    try:
        search = await asyncio.to_thread(_ddg_search, query)
        searches_list = [{"title": i.get("title", ""), 
                          "href": i["href"]} 
                          for i in search if i.get("href")]
        texts = await asyncio.gather(*[
            asyncio.to_thread(_fetch_and_extract, item['href'])
            for item in search
            if item.get('href')
        ])
        combined = '\n\n'.join(t for t in texts if t)
        if not combined:
            return "No text were extracted", []
        return await generate_summary(combined), searches_list
    except DDGSException as e:
        return f'ddgs search error {e}', []
    except Exception as e:
        return f"Some system error: {e}", []