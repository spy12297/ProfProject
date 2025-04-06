import streamlit as st
from document_search import DocumentSearch
from pathlib import Path
import json
import time

st.set_page_config(
    page_title="Умный поиск по документам",
    page_icon="🔍",
    layout="wide"
)

def initialize_search_engine():
    if 'search_engine' not in st.session_state:
        try:
            st.session_state.search_engine = DocumentSearch()
        except Exception as e:
            st.error(f"Ошибка инициализации поисковой системы: {str(e)}")
            st.stop()

def display_system_info():
    """Отображение информации о системе"""
    with st.sidebar:
        st.header("Системная информация")
        if hasattr(st.session_state.search_engine, 'components'):
            st.write("Поддерживаемые форматы:")
            components = st.session_state.search_engine.components
            
            formats = {
                'docx': ('Word (.docx)', '📄'),
                'xlsx': ('Excel (.xlsx)', '📊'),
                'pptx': ('PowerPoint (.pptx)', '📑'),
                'msg': ('Outlook (.msg)', '📧')
            }
            
            for fmt, (name, icon) in formats.items():
                if components.get(fmt):
                    st.success(f"{icon} {name}")
                else:
                    st.error(f"{icon} {name} (не поддерживается)")
            
            if components.get('embeddings'):
                st.success(f"🧠 Используется {st.session_state.search_engine.device}")
            else:
                st.error("🧠 Векторные вычисления недоступны")

def main():
    st.title("🔍 Умный поиск по документам")
    
    initialize_search_engine()
    display_system_info()
    
    # Основная область
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.info(
            "База данных документов:\n"
            f"{st.session_state.search_engine.base_path}"
        )
        
        if st.button("🔄 Переиндексировать документы", type="primary"):
            with st.spinner("Идёт индексация документов..."):
                try:
                    start_time = time.time()
                    stats = st.session_state.search_engine.index_documents()
                    duration = time.time() - start_time
                    
                    if stats['indexed_files'] > 0:
                        st.success(
                            f"✅ Индексация завершена за {duration:.1f} сек!\n\n"
                            f"📊 Статистика:\n"
                            f"- Обработано файлов: {stats['processed_files']}\n"
                            f"- Проиндексировано: {stats['indexed_files']}\n"
                            f"- Пропущено: {stats['skipped_files']}"
                        )
                        if stats['errors']:
                            with st.expander("🚫 Ошибки при индексации"):
                                for error in stats['errors']:
                                    st.error(error)
                    else:
                        st.warning("⚠️ Нет файлов для индексации")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при индексации: {str(e)}")
    
    with col1:
        search_query = st.text_input(
            "Поисковый запрос:",
            placeholder="Например: информация о внедрении системы Поток",
            help="Введите текст для поиска по документам"
        )
        
        col3, col4 = st.columns([1, 4])
        with col3:
            n_results = st.number_input(
                "Количество результатов",
                min_value=1,
                max_value=20,
                value=5
            )
        
        if st.button("🔎 Искать", type="primary"):
            if search_query:
                with st.spinner("🔍 Выполняется поиск..."):
                    results = st.session_state.search_engine.search(search_query, n_results)
                    
                    if results:
                        st.subheader(f"📑 Найдено {len(results)} документов:")
                        
                        # Создаем архив с результатами
                        archive_path = st.session_state.search_engine.create_results_archive(results)
                        
                        for i, result in enumerate(results, 1):
                            score_color = "green" if result['relevance_score'] > 0.7 else "orange"
                            with st.expander(
                                f"{i}. {result['file_name']} "
                                f"(Релевантность: :{score_color}[{result['relevance_score']:.2f}])"
                            ):
                                st.text("📍 Путь к файлу:")
                                st.code(result['file_path'])
                                st.text("📝 Фрагмент документа:")
                                st.markdown(result['snippet'])
                        
                        if archive_path:
                            with open(archive_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Скачать найденные документы",
                                    data=f,
                                    file_name=Path(archive_path).name,
                                    mime="application/zip",
                                    help="Скачать архив со всеми найденными документами"
                                )
                    else:
                        st.warning("🔍 По вашему запросу ничего не найдено.")
            else:
                st.warning("⚠️ Введите поисковый запрос.")

if __name__ == "__main__":
    main()