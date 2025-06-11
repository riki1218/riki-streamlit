import streamlit as st
import numpy as np
import pandas as pd

st.title("Streamlit　お試し")

st.write("何かテキスト追加")

df=pd.DataFrame({
    "1列目":[1,2,3,4],
    "2列目":[10,20,30,40]
})

st.dataframe(df.style.highlight_max(axis=0))

"""
# 章
## 節
### 項

```python
importstreamlit as st
```
"""

df=pd.DataFrame(
    np.random.rand(20,3),
   columns=["a","b","c"]
)
st.line_chart(df)


df=pd.DataFrame(
    np.random.rand(200,2)/[50,50]+[35.69,139.70],
   columns=["lat","lon"]
)
st.map(df)


from PIL import Image


if st.checkbox("Show Image"):
    st.write("画像を表示")
    image_path = "C:/Users/ricky/Desktop/写真ｆ１/wallpaper-1280x720.jpeg"
    img=Image.open(image_path)
    st.image(img,caption="MercedesF1",use_container_width=True)


condistion=st.sidebar.slider("あなたの今の調子は？",0,100,10)
"コンディション：",condistion

left_column,right_column=st.columns(2)
button=left_column.button("右に画像を表示")
if button:
    right_column.write("ここは右")

expander1=st.expander("問い合わせ")
expander1.write("回答")

import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 環境変数の設定 ---
os.environ["USER_AGENT"] = "MyLangChainBot/1.0"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA7ff88iym-pPHbjSKf-x6ddwFYzULZ9ec"  # セキュリティ上は .env 推奨

# --- 初回読み込み時のみドキュメント処理 ---
@st.cache_resource
def load_vectorstore():
    loader = WebBaseLoader(web_paths=("https://news.yahoo.co.jp/",))
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    return vectorstore

# --- RAG チェーン構築 ---
@st.cache_resource
def build_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_template("""
    あなたはユーザーの質問に答えるAIアシスタントです。
    提供された情報のみに基づいて、質問に答えてください。
    情報が不足している場合は、その旨を伝えてください。

    {context}

    質問: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# --- Streamlit UI ---
st.title("📰 今日のニュース応答LLM（RAG × Gemini）")
question = st.text_input("質問を入力してください:", "今日のニュースを教えてください。")

if st.button("質問する"):
    with st.spinner("情報を検索して回答を生成しています..."):
        rag_chain = build_rag_chain()
        response = rag_chain.invoke(question)
        st.markdown("### 回答")
        st.write(response)



st.title("💰 年間売上推移グラフ (2018-2021)")

st.write("2018年から2021年までの指定された売上データに基づき、折れ線グラフを表示します。")

# --- 売上データのDataFrameを作成 ---
# 年とそれぞれの売上データを定義
years = [2018,2019,2020,2021,2022,2023,2024,2025]
sales_million_yen = [2844447, 2913446, 3095234, 2994023, 3014095, 3313018, 3477262, 3423431] # 単位：百万円

# DataFrameを作成
# '年' 列をインデックスに設定すると、st.line_chart が自動的にX軸として認識します
df_yearly_sales = pd.DataFrame({
    '年': years,
    '売上 (百万円)': sales_million_yen
})


# ✅ 修正点1: '年' 列をインデックスに設定し、さらにそのインデックスを整数型に変換
df_yearly_sales = df_yearly_sales.set_index('年').astype(int)

st.subheader("📊 年間売上データ")
# 作成したDataFrameを表示
st.dataframe(df_yearly_sales)

# --- 折れ線グラフの表示 ---
st.subheader("📈 売上推移グラフ")
# DataFrameをst.line_chartに渡してグラフを表示
st.line_chart(df_yearly_sales)

st.markdown("---")
st.caption("これは指定されたデータに基づいた売上推移の例です。")




import altair as alt # Altair をインポート

st.title("📈 年ごとの売上推移グラフ (ラベル追加版)")

st.write("2018年から2025年までの指定された売上データを折れ線グラフで表示します。")

# --- 1. 売上データのDataFrameを作成する ---
sales_data = {
    '年': [2018,2019,2020,2021,2022,2023,2024,2025],
    '売上 (百万円)': [2844447, 2913446, 3095234, 2994023, 3014095, 3313018, 3477262, 3423431]
}

df_yearly_sales = pd.DataFrame(sales_data)

st.subheader("📊 年間売上データ")
st.dataframe(df_yearly_sales)

# --- 2. 折れ線グラフを表示する (Altair を直接使用) ---
st.subheader("📈 年間売上推移グラフ")

# AltairのChartオブジェクトを構築
chart = alt.Chart(df_yearly_sales).mark_line(point=True).encode( # point=Trueで各データポイントに点を表示
    # ✅ 修正点: X軸にタイトル（ラベル）を追加し、フォーマットを整数に指定
    x=alt.X('年', axis=alt.Axis(format='d', title='年度')), # X軸ラベルを「年度」に設定

    # ✅ 修正点: Y軸にタイトル（ラベル）を追加
    y=alt.Y('売上 (百万円)', title='売上 (百万円)'), # Y軸ラベルを「売上 (万円)」に設定
    tooltip=['年', '売上 (百万円)'] # ホバー時にツールチップを表示
).properties(
    title='年間売上推移' # グラフ全体のタイトル
).interactive() # ズームやパンを可能にする

st.altair_chart(chart, use_container_width=True) # AltairチャートをStreamlitで表示

st.markdown("---")
st.caption("このグラフは指定された売上データに基づいています。")


