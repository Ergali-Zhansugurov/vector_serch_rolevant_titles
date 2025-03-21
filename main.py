import os
import json
from infrastructure.embedding import get_embedding, create_index_text
from infrastructure.vector_index import VectorIndex, save_index, load_index
from presentation.telegram_bot import run_bot

def build_index_from_file(data_file: str, index_file: str, dimension: int = 768) -> VectorIndex:
    # Если файл индекса существует, загрузим его
    if os.path.exists(index_file):
        print(f"Загрузка индекса из файла {index_file}")
        index = load_index(index_file, dimension)
        return index
    else:
        print(f"Создание нового индекса и сохранение в файл {index_file}")
        index = VectorIndex(dimension)
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for manga in data:
            text_for_embedding = create_index_text(manga)
            emb = get_embedding(text_for_embedding)
            index.add_item(emb, manga)
        index.build_index()
        save_index(index, index_file)
        return index

def main():
    data_file = "updated_all_mangas.json"
    index_file = "fias_index.bin"
    # Если нужно обновить формат эмбеддингов, удалите файл index_file
    index = build_index_from_file(data_file, index_file)
    print("Индекс готов. Запуск Telegram-бота.")
    run_bot(index)

if __name__ == "__main__":
    main()
