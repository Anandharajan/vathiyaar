def test_imports():
    import app.asr  # noqa: F401
    import app.config  # noqa: F401
    import app.llm  # noqa: F401
    import app.main  # noqa: F401
    import app.memory  # noqa: F401
    import app.prompts  # noqa: F401
    import app.rag  # noqa: F401
    import app.tools  # noqa: F401
    import app.tts  # noqa: F401

    assert True
