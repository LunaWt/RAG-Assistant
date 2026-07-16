import pytest
import app.services.tools as tools_module
from ddgs.exceptions import DDGSException

from app.services.tools import calculator, search_knowledge_base, web_search


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 2", 4),
        ("4 * 2 / 2 + 4", 8),
        ("1/4 * 1/2", 0.125),
        ("0.15389 + 0.001", 0.1549),
    ],
)
def test_calculator_valid_expressions(expression, expected):
    result = calculator(expression)
    assert isinstance(result, (int, float))
    assert result == expected


@pytest.mark.parametrize(
    "expression, expected_prefix",
    [
        ("0" * 101, "Слишком большое выражение"),
        ("1 +", "Синтаксическая ошибка в выражении"),
        ("foo(2)", "Использование запрещенных переменных или функций"),
        ("x + 1", "Использование запрещенных переменных или функций"),
        (1, "Выражение должно быть строкой"),
        ("2 ** 2000", "Ошибка в выражении"),
    ],
)
def test_calculator_error_messages(expression, expected_prefix):
    assert str(calculator(expression)).startswith(expected_prefix)


def test_search_knowledge_base_without_filename(monkeypatch: pytest.MonkeyPatch):
    def fake_rag_search(query: str) -> list[str]:
        return [query, "Информация про Калифорнию.", "Информация про нейронные сети"]

    monkeypatch.setattr(
        tools_module.vector_db,
        "rag_search",
        fake_rag_search,
    )

    result = search_knowledge_base(query="California search")

    assert result == (
        "California search\n\n"
        "Информация про Калифорнию.\n\n"
        "Информация про нейронные сети"
    )


def test_search_knowledge_base_with_filename(monkeypatch: pytest.MonkeyPatch):
    def fake_rag_search_with_filename(query: str, filename: str) -> list[str]:
        return [query, filename, "Информация про Калифорнию."]

    monkeypatch.setattr(
        tools_module.vector_db,
        "rag_search",
        fake_rag_search_with_filename,
    )

    result = search_knowledge_base(
        query="California search", filename="File about California"
    )

    assert result == (
        "California search\n\nFile about California\n\nИнформация про Калифорнию."
    )


def test_search_knowledge_base_error(monkeypatch: pytest.MonkeyPatch):
    def fake_rag_search_error(query: str) -> list[str]:
        raise RuntimeError("database error")

    monkeypatch.setattr(
        tools_module.vector_db,
        "rag_search",
        fake_rag_search_error,
    )

    result = search_knowledge_base(
        query="California search",
    )

    assert (
        result
        == "Knowledge base search failed: database error. Try web_search or answer from your own knowledge."
    )


@pytest.mark.asyncio
async def test_web_search_success(monkeypatch: pytest.MonkeyPatch):
    california_text = (
        "Климат Калифорнии невероятно разнообразен "
        "и во многом зависит от удаленности региона от океана и гор"
    )
    san_francisco_text = (
        "Погода в Сан-Франциско, в отличие от остальной Калифорнии, "
        "круглый год остается прохладной и сильно зависит от влияния Тихого океана"
    )
    searches_list_dict = [
        {
            "title": "California_weather",
            "href": "https://Cali_Weather.com",
        },
        {
            "title": "San_Fransisco_weather",
            "href": "https://San_Weather.com",
        },
    ]

    def fake_ddg_search(query: str) -> list[dict]:
        return [
            {
                "title": "California_weather",
                "href": "https://Cali_Weather.com",
                "body": "Whatever..........",
            },
            {
                "title": "San_Fransisco_weather",
                "href": "https://San_Weather.com",
                "body": "Whatever..........",
            },
        ]

    def fake_fetch_and_extract(url: str) -> str | None:
        if url == "https://Cali_Weather.com":
            return california_text

        if url == "https://San_Weather.com":
            return san_francisco_text

    async def fake_generate_summary(content: str) -> str:
        assert content == f"{california_text}\n\n{san_francisco_text}"
        return """Климат Калифорнии в основном средиземноморский 
        с жарким засушливым летом и мягкой зимой, 
        однако погода сильно контрастирует между прохладным 
        океанским побережьем и раскаленными внутренними долинами. 
        На этом фоне Сан-Франциско выделяется уникальными условиями: 
        из-за влияния океана и знаменитых густых туманов 
        здесь круглый год царит прохлада, 
        а самыми теплыми месяцами становятся не летние, 
        а сентябрь и октябрь [Data Commons, NCEI]."""

    monkeypatch.setattr(tools_module, "_ddg_search", fake_ddg_search)
    monkeypatch.setattr(tools_module, "_fetch_and_extract", fake_fetch_and_extract)
    monkeypatch.setattr(tools_module, "generate_summary", fake_generate_summary)

    result, searches_list = await web_search("Погода в калифорнии")

    expected_text = """Климат Калифорнии в основном средиземноморский 
        с жарким засушливым летом и мягкой зимой, 
        однако погода сильно контрастирует между прохладным 
        океанским побережьем и раскаленными внутренними долинами. 
        На этом фоне Сан-Франциско выделяется уникальными условиями: 
        из-за влияния океана и знаменитых густых туманов 
        здесь круглый год царит прохлада, 
        а самыми теплыми месяцами становятся не летние, 
        а сентябрь и октябрь [Data Commons, NCEI]."""

    assert result == expected_text
    assert searches_list == searches_list_dict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_results",
    [
        [],
        [{"title": "Empty page", "href": "https://example.com/empty"}],
    ],
)
async def test_web_search_returns_no_text_when_nothing_is_extracted(
    monkeypatch: pytest.MonkeyPatch, search_results: list[dict]
) -> None:
    fetched_urls: list[str] = []

    def fake_ddg_search(query: str) -> list[dict]:
        return search_results

    def fake_fetch_and_extract(url: str) -> None:
        fetched_urls.append(url)
        return None

    monkeypatch.setattr(tools_module, "_ddg_search", fake_ddg_search)
    monkeypatch.setattr(tools_module, "_fetch_and_extract", fake_fetch_and_extract)

    assert await web_search("query") == ("No text were extracted", [])
    expected_urls = [item["href"] for item in search_results if item.get("href")]
    assert fetched_urls == expected_urls


@pytest.mark.asyncio
async def test_web_search_omits_results_without_href(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched_urls: list[str] = []

    def fake_ddg_search(query: str) -> list[dict]:
        return [
            {"title": "No link"},
            {"title": "Empty link", "href": ""},
            {"title": "Valid result", "href": "https://example.com/valid"},
        ]

    def fake_fetch_and_extract(url: str) -> str:
        fetched_urls.append(url)
        return "Valid page text"

    async def fake_generate_summary(content: str) -> str:
        assert content == "Valid page text"
        return "Summary"

    monkeypatch.setattr(tools_module, "_ddg_search", fake_ddg_search)
    monkeypatch.setattr(tools_module, "_fetch_and_extract", fake_fetch_and_extract)
    monkeypatch.setattr(tools_module, "generate_summary", fake_generate_summary)

    assert await web_search("query") == (
        "Summary",
        [{"title": "Valid result", "href": "https://example.com/valid"}],
    )
    assert fetched_urls == ["https://example.com/valid"]


@pytest.mark.asyncio
async def test_web_search_summarizes_non_empty_text_and_keeps_valid_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_results = [
        {"title": "First", "href": "https://example.com/first"},
        {"title": "No text", "href": "https://example.com/none"},
        {"title": "Second", "href": "https://example.com/second"},
    ]

    def fake_ddg_search(query: str) -> list[dict]:
        return search_results

    def fake_fetch_and_extract(url: str) -> str | None:
        return {
            "https://example.com/first": "First extracted text",
            "https://example.com/none": None,
            "https://example.com/second": "Second extracted text",
        }[url]

    async def fake_generate_summary(content: str) -> str:
        assert content == "First extracted text\n\nSecond extracted text"
        return "Combined summary"

    monkeypatch.setattr(tools_module, "_ddg_search", fake_ddg_search)
    monkeypatch.setattr(tools_module, "_fetch_and_extract", fake_fetch_and_extract)
    monkeypatch.setattr(tools_module, "generate_summary", fake_generate_summary)

    assert await web_search("query") == (
        "Combined summary",
        [
            {"title": "First", "href": "https://example.com/first"},
            {"title": "No text", "href": "https://example.com/none"},
            {"title": "Second", "href": "https://example.com/second"},
        ],
    )


@pytest.mark.asyncio
async def test_web_search_returns_ddgs_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ddg_search(query: str) -> list[dict]:
        raise DDGSException("service unavailable")

    monkeypatch.setattr(tools_module, "_ddg_search", fake_ddg_search)

    assert await web_search("query") == ("ddgs search error service unavailable", [])


@pytest.mark.asyncio
async def test_web_search_returns_system_error_for_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ddg_search(query: str) -> list[dict]:
        raise RuntimeError("dependency failed")

    monkeypatch.setattr(tools_module, "_ddg_search", fake_ddg_search)

    assert await web_search("query") == ("Some system error: dependency failed", [])
