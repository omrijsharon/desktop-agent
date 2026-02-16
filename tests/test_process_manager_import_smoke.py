def test_process_manager_import_smoke() -> None:
    from desktop_agent.process_manager import main

    assert callable(main)

