import pytest

from fastapi.testclient import TestClient

import app.main as main_module

def test_list_document_sources(monkeypatch: pytest.MonkeyPatch):
    def fake_list_sources():
        return ['Погода в калифорнии', 'Книга о Python']
    
    monkeypatch.setattr(
        main_module.vector_db,
        'list_sources',
        fake_list_sources,
    )
    
    client = TestClient(main_module.app)
    res = client.get('/documents/sources')

    assert res.status_code == 200
    assert res.json() == {
        "sources": ['Погода в калифорнии', 'Книга о Python']
        }









