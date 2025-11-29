import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# --- 1. Настройки страницы (Белая тема) ---
st.set_page_config(page_title="Пиксель", page_icon="✨", layout="centered")

# --- 2. CSS: Простой и чистый дизайн ---
st.markdown("""
<style>
    /* Основной фон белый */
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Заголовок */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica', sans-serif;
    }

    /* Стиль кнопки */
    .stButton > button {
        background-color: #3498db; /* Спокойный синий */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }

    /* Блок рассуждений (Светло-серый, технический) */
    .reasoning-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        font-size: 0.9em;
        color: #666;
        margin-bottom: 20px;
        font-family: monospace;
    }

    /* Блок ответа (Акцентный, красивый) */
    .answer-box {
        background-color: #ffffff;
        border-left: 5px solid #3498db; /* Синяя линия слева */
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        font-size: 1.1em;
        line-height: 1.6;
        color: #2c3e50;
    }
    
    /* Заголовки внутри блоков */
    .box-title {
        font-weight: bold;
        margin-bottom: 10px;
        display: block;
        text-transform: uppercase;
        font-size: 0.8em;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Инициализация ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data
def create_knowledge_base():
    try:
        # Загружаем данные
        works_df = pd.read_csv("ПроизведенияП.csv").astype(str).fillna('не указано')
        return works_df
    except Exception as e:
        st.error(f"Ошибка при чтении файла 'ПроизведенияП.csv': {e}")
        return None

# --- 4. Интерфейс ---
st.title("✨ Умный ассистент Пиксель")
st.caption("Задайте вопрос о фильмах и мультфильмах Disney")

# Поле ввода
user_query = st.text_input("Ваш вопрос:", placeholder="Например: Какие мультфильмы вышли в 2010 году?")
ask_button = st.button("Найти ответ")

# Загрузка базы
works_df = create_knowledge_base()
answer_placeholder = st.empty()

# --- 5. Логика обработки ---
if works_df is not None and GROQ_API_KEY:
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct" # Или "llama3-70b-8192"
    except Exception as e:
        st.error(f"Ошибка клиента: {e}")
        client = None

    if client and user_query and ask_button:
        with st.spinner("Анализирую базу данных..."):
            try:
                # Сборка контекста из базы
                knowledge_text = ""
                for _, work in works_df.iterrows():
                    knowledge_text += "---\n"
                    knowledge_text += f"Название: {work.get('Name')}\n"
                    knowledge_text += f"Тип: {work.get('Тип')} (Важно: Фильм или Мультфильм)\n"
                    knowledge_text += f"Год: {work.get('Год выпуска')}\n"
                    knowledge_text += f"Жанр: {work.get('Жанр')}\n"
                    knowledge_text += f"Рейтинг: {work.get('Рейтинг')}\n"
                    knowledge_text += f"Сюжет: {work.get('Описание')}\n"
                    knowledge_text += f"Персонажи: {work.get('Персонажи')}\n"

                # Промпт с жесткими правилами
                prompt = f"""
                Твоя роль - эксперт по базе данных Disney.
                
                ИНСТРУКЦИИ:
                1. Отвечай ТОЛЬКО на основе предоставленных данных.
                2. СТРОГО различай типы: "Фильм" (живые актеры) и "Мультфильм" (анимация).
                   - Если спрашивают про ФИЛЬМЫ -> игнорируй мультфильмы.
                   - Если спрашивают про МУЛЬТФИЛЬМЫ -> игнорируй фильмы.
                3. Если данных нет, ответь: "Информации нет в базе".

                ФОРМАТ ОТВЕТА (ОБЯЗАТЕЛЬНО):
                Ты должен вернуть ответ строго в двух блоках.

                [РАССУЖДЕНИЯ]
                Здесь опиши ход поиска. Какие записи нашел? Какой у них "Тип"? Подходят ли они под год/жанр из вопроса?
                Пример: "Нашел запись 'Король Лев', тип Мультфильм, год 1994. Подходит под запрос."

                [ОТВЕТ]
                Здесь напиши красивый финальный ответ для пользователя. Без технических деталей, только суть.

                ДАННЫЕ:
                {knowledge_text}

                ВОПРОС: {user_query}
                """

                # Запрос к нейросети
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                answer_content = response.choices[0].message.content

                # Разделение ответа на блоки
                try:
                    parts = answer_content.split("[ОТВЕТ]")
                    reasoning = parts[0].replace("[РАССУЖДЕНИЯ]", "").strip()
                    final_answer = parts[1].strip()
                except:
                    # Если модель ошиблась с форматом, выводим как есть
                    reasoning = "Модель не предоставила рассуждения в нужном формате."
                    final_answer = answer_content.replace("[РАССУЖДЕНИЯ]", "").replace("[ОТВЕТ]", "")

                # Превращаем переносы строк в <br> для HTML
                reasoning_html = reasoning.replace('\n', '<br>')
                final_answer_html = final_answer.replace('\n', '<br>')

                # Вывод на экран
                st.markdown(f"""
                    <div class="reasoning-box">
                        <span class="box-title">⚙️ Логика поиска (Рассуждения):</span>
                        {reasoning_html}
                    </div>
                    
                    <div class="answer-box">
                        <span class="box-title">📝 Ответ:</span>
                        {final_answer_html}
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")

    elif not user_query and ask_button:
        st.warning("Пожалуйста, напишите вопрос.")

elif not works_df:
    st.error("Файл 'ПроизведенияП.csv' не найден или пуст.")
elif not GROQ_API_KEY:
    st.error("API ключ не найден.")
