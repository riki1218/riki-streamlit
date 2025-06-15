import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

import time

progress_text = "Operation in progress. 🚀読み込み中."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

st.button("押下で更新")

st.title("NECの関連データとLLM(RAG) デモページ")



st.badge("2025/6/15更新", icon=":material/check:", color="green")

st.subheader(":blue[2024年度] 経営指標と前年比", divider=True)


col1, col2, col3 = st.columns(3)
col1.metric("売上高", "3兆4,234億円", "-1.5 %")
col2.metric("売上収益当期利益率", "5.1%", "0.8%")
col3.metric("従業員数(連結)", "104,194名", "-1,082名")

st.write("")
st.page_link("https://jpn.nec.com/ir/library/securities.html?", label="有価証券報告書のリンクはこちら", icon="🔗")

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

import altair as alt # Altair をインポート

st.title("年ごとのＮＥＣ売上推移グラフ")

st.write("2018年から2025年までの指定された売上データを折れ線グラフで表示します。")

# --- 1. 売上データのDataFrameを作成する ---
sales_data = {
    '年': [2018,2019,2020,2021,2022,2023,2024,2025],
    '売上 (百万円)': [2844447, 2913446, 3095234, 2994023, 3014095, 3313018, 3477262, 3423431]
}

df_yearly_sales = pd.DataFrame(sales_data)



# --- 2. 折れ線グラフを表示する (Altair を直接使用) ---
st.header("📈 年間売上推移グラフ")

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


st.subheader("📊 年間売上データ")
st.dataframe(df_yearly_sales,height=320, width=600)



st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")


import plotly.express as px


st.title("事業別売上高推移グラフ")

# --- 1. 提供されたサンプルデータの作成 ---
df_original = pd.DataFrame(
    {
        "name": ["社会公共事業", "社会基盤事業", "エンタープライズ事業", "ネットワークサービス事業", "グローバル事業"],
        "売上高": [
            [268337, 286151, 478352, 425060, 442637, 456687],
            [624819, 621879, 678767, 692876, 608413, 649662],
            [405221, 431801, 549796, 503074, 574680, 614369],
            [442472, 460307, 482692, 538810, 511547, 543400],
            [420451, 409369, 493073, 449988, 485578, 586336],
        ]
    }
)

# --- 2. データをPlotlyが扱いやすい「長い形式（long format）」に変換 ---
# ここでは、簡略化のためインデックスを「時間軸」として扱います。


data_for_plot = []
for index, row in df_original.iterrows():
    business_name = row['name']
    for i, views in enumerate(row['売上高']):
        data_for_plot.append({
            '事業名': business_name,
            '時点': i + 2018,  # 1から始まる時点（例: 1ヶ月目, 2ヶ月目など）
            '売上高': views
        })

df_plot = pd.DataFrame(data_for_plot)


st.header("📈 事業別売上高推移")

# --- 3. フィルターの設置 ---
# グラフに表示する事業名の選択肢を取得
all_business_names = df_plot['事業名'].unique().tolist()

# 初期選択として全ての事業名を選択
default_selected_names = all_business_names

# マルチセレクトボックスで表示する事業名を選択
selected_names = st.multiselect(
    "表示する事業を選択してください:",
    options=all_business_names,
    default=default_selected_names # 初期表示で全ての線を選択
)

# 選択された事業が存在しない場合の処理
if not selected_names:
    st.warning("表示する事業が選択されていません。少なくとも1つ選択してください。")
else:
    # --- 4. 選択されたデータでグラフを作成 ---
    # 選択された事業名のみをフィルター
    df_filtered = df_plot[df_plot['事業名'].isin(selected_names)]

    # Plotly Expressで折れ線グラフを作成
    fig = px.line(
        df_filtered,
        x='時点',        # X軸に「時点」を使用
        y='売上高',    # Y軸に「売上高」を使用
        color='事業名',  # 「事業名」によって線が色分けされ、凡例が作成されます
        title="選択された事業の売上高推移",
        labels={
            '時点': '決算年', # より分かりやすいラベルに変更
            '売上高': '売上高',
            '事業名': '事業区分'
        },
        markers=True # 各データ点にマーカーを表示
    )

    # レイアウトの調整（オプション）
    fig.update_layout(
        hovermode="x unified", # ホバー時にx軸に沿って複数の系列の値を表示
        xaxis_title="決算年",
        yaxis_title="売上高",
        legend_title="事業区分" # 凡例のタイトルを設定
    )

    # Y軸のフォーマットを整数に設定（必要に応じて）
    fig.update_yaxes(tickformat=".0f")

    # --- 5. Streamlitでグラフを表示 ---
    st.plotly_chart(fig, use_container_width=True) # コンテナの幅に合わせて表示


# st.markdown("---")
# st.markdown("### 元データ (変換前)")
# st.dataframe(df_original)

# st.markdown("### グラフ用データ (変換後の一部)")
# st.dataframe(df_plot.head())


st.subheader("📊 事業別年間売上データ")
df = pd.DataFrame(
    {
        "name": ["社会公共事業", "社会基盤事業", "エンタープライズ事業","ネットワークサービス事業","グローバル事業"],
        "url": ["https://jpn.nec.com/ir/pdf/library/220907/public_solutions.pdf",
                 "https://jpn.nec.com/ir/pdf/library/210915/public_infrastructure.pdf",
                   "https://jpn.nec.com/ir/pdf/library/190716/enterprise.pdf",
                   "https://jpn.nec.com/ir/pdf/library/190716/network_services.pdf",
                   "https://jpn.nec.com/ir/pdf/library/220907/global.pdf"
                ],
        "views_history": [
            [268337,286151,478352,425060,442637,456687],
            [624819,621879,678767,692876,608413,649662],
            [405221,431801,549796,503074,574680,614369],
            [442472,460307,482692,538810,511547,543400],
            [420451,409369,493073,449988,485578,586336],
            ]
    }
)
st.dataframe(
    df,
    column_config={
        "name": "セグメント別",
        "url": st.column_config.LinkColumn("URL"),
        "views_history": st.column_config.LineChartColumn(
            "2018~2023の決算データ", y_min=268337, y_max=692876
        ),
    },
    hide_index=True,
)


st.markdown("---")
st.caption("このグラフ・データはＨＰの財務業績データページに基づいて作成。")

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

st.header("🚀本社地区について")

from PIL import Image


if st.checkbox("チェックで本社の画像を標示"):
    st.write("ロケットタワー")
    image_path = "C:/Users/ricky/Pictures/th_IMG_6341.jpg"
    img=Image.open(image_path)
    st.image(img,caption="ロケットタワー",use_container_width=True)


"""

### 拠点マップ

"""

df=pd.DataFrame(
    [[35.649442, 139.747894],
     [35.65134063, 139.7486096],
     [35.650659,139.75416],
     [35.650354,139.746775],
     [35.648491,139.747165]
    ],
   columns=["lat","lon"]
)
st.map(df)


condistion=st.sidebar.slider("NECの関東の拠点の数は？",0,30,5)
st.sidebar.write("あなたの回答：",condistion)


expander1=st.sidebar.expander("関東の拠点数 回答")
expander1.write("HPによると12だが絶対もっとある")


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")


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
st.title("📰 NEC応答LLM（RAG × Gemini）")
question = st.text_input("質問を入力してください:", "NECのニュースを教えてください。")

if st.button("質問する"):
    with st.spinner("情報を検索して回答を生成しています..."):
        rag_chain = build_rag_chain()
        response = rag_chain.invoke(question)
        st.markdown("### 回答")
        st.write(response)


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")




def show_references_text():
    st.header("参考資料")

    st.write("""
    **書籍:**
    * [1] 末次拓斗著, 「誰でもわかる大規模言語モデル入門」, 2024年.
    * [2] 梅田弘之著 , 「エンジニアなら知っておきたいAIのキホン」, 2019年.

    **Webサイト:**
    * [3] Streamlit公式ドキュメント: [https://docs.streamlit.io/](https://docs.streamlit.io/)
    * [4] NEC公式HP: [https://jpn.nec.com/profile/corp/profile.html](https://jpn.nec.com/profile/corp/profile.html)

    **共著:**
    * [5] Gemini 2.5 Flash君
    """)

if __name__ == "__main__":
    show_references_text()



st.write("")
st.write("")
st.write("")
st.write("")


sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
selected = st.feedback("thumbs")
if selected is not None:
    st.markdown(f"You selected: {sentiment_mapping[selected]} , and Riki is satisfied.")





