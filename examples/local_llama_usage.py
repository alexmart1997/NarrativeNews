from __future__ import annotations

from app.config.settings import get_settings
from app.services import create_llm_client


def main() -> None:
    settings = get_settings()
    client = create_llm_client(settings)
    if client is None:
        print("LLM provider is disabled.")
        return

    text = client.generate_text(
        "Скажи одной короткой фразой, что локальный LLM client подключен.",
        system_prompt="Отвечай по-русски и очень кратко.",
        max_tokens=40,
    )
    print(text)


if __name__ == "__main__":
    main()
