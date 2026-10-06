import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

st.set_page_config(page_title="Acompanhamento de Semaglutida", page_icon="📉", layout="centered")

CLASSES_VARIACAO_PESO = ("Perda", "Estável", "Ganho")
LIMIAR_VARIACAO_PESO_KG = 0.1
MIN_REGISTROS_TREINO_ML = 4
MIN_REGISTROS_AVALIACAO_ML = 6

# --- 1. CONFIGURAÇÃO DO SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except KeyError as error:
    raise RuntimeError(
        "Configure SUPABASE_URL e SUPABASE_KEY em .streamlit/secrets.toml."
    ) from error

APP_URL = st.secrets.get("APP_URL", "").strip()
LOGO_PATH = Path(__file__).with_name("LOGO-IMAGE-2026.png")


class StreamlitAuthStorage:
    def get_item(self, key):
        return st.session_state.get(f"supabase_auth_{key}")

    def set_item(self, key, value):
        st.session_state[f"supabase_auth_{key}"] = value
        if key.endswith("-code-verifier"):
            st.session_state.supabase_pkce_verifier = value

    def remove_item(self, key):
        st.session_state.pop(f"supabase_auth_{key}", None)


def get_supabase_client():
    if "supabase_client" not in st.session_state:
        options = SyncClientOptions(
            storage=StreamlitAuthStorage(),
            flow_type="pkce",
        )
        st.session_state.supabase_client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options,
        )
    return st.session_state.supabase_client

supabase = get_supabase_client()


def get_authenticated_user():
    auth_code = st.query_params.get("code")
    if auth_code:
        code_verifier = st.query_params.get("pkce_verifier")
        exchange_params = {"auth_code": auth_code}
        if code_verifier:
            exchange_params["code_verifier"] = code_verifier
        try:
            supabase.auth.exchange_code_for_session(exchange_params)
            st.query_params.clear()
        except Exception as error:
            if "code verifier" in str(error).lower():
                    st.error("A sessão de login expirou. Clique em Entrar com Google novamente.")
            else:
                st.error("Não foi possível concluir o login com Google. Tente novamente.")
            st.stop()

    session = supabase.auth.get_session()
    return session.user if session else None


def login_with_google():
    if not APP_URL:
        st.error("Configure APP_URL com a URL pública do Streamlit Cloud nos secrets.")
        st.stop()

    credentials = {"provider": "google"}
    credentials["options"] = {"redirect_to": APP_URL}
    try:
        response = supabase.auth.sign_in_with_oauth(credentials)
    except Exception as error:
        if "provider is not enabled" in str(error).lower():
            st.error(
                "O login Google ainda não está ativado no Supabase. "
                "Ative Authentication > Providers > Google e configure as credenciais OAuth."
            )
            st.stop()
        raise
    code_verifier = st.session_state.get("supabase_pkce_verifier")
    if not code_verifier:
        st.error("Não foi possível preparar a sessão segura do login. Tente novamente.")
        st.stop()

    authorization_url = urlsplit(response.url)
    authorization_query = parse_qs(authorization_url.query, keep_blank_values=True)
    redirect_url = authorization_query.get("redirect_to", [APP_URL])[0]
    redirect_parts = urlsplit(redirect_url)
    redirect_query = parse_qs(redirect_parts.query, keep_blank_values=True)
    redirect_query["pkce_verifier"] = [code_verifier]
    redirect_url = urlunsplit((
        redirect_parts.scheme,
        redirect_parts.netloc,
        redirect_parts.path,
        urlencode(redirect_query, doseq=True),
        redirect_parts.fragment,
    ))
    authorization_query["redirect_to"] = [redirect_url]
    authorization_url = urlunsplit((
        authorization_url.scheme,
        authorization_url.netloc,
        authorization_url.path,
        urlencode(authorization_query, doseq=True),
        authorization_url.fragment,
    ))
    st.link_button("Entrar com Google", authorization_url, use_container_width=True)

# --- 2. FUNÇÕES DE BASE DE DADOS ---
def obter_perfil(user_id):
    try:
        resposta = supabase.table('utilizadores').select('*').eq('id', user_id).execute()
    except Exception:
        st.error("Não foi possível consultar o perfil no Supabase.")
        st.info(
            "No SQL Editor do Supabase, confirme que a tabela public.utilizadores "
            "existe e execute database/migrate_google_auth.sql."
        )
        st.stop()
    return resposta.data[0] if resposta.data else None

def obter_historico(user_id):
    resposta = supabase.table('registos_diarios').select('*').eq('user_id', user_id).order('data_registo').execute()
    return pd.DataFrame(resposta.data)

def guardar_registo(user_id, data_registo, peso, tomou, dose):
    try:
        supabase.table('registos_diarios').upsert({
            'user_id': user_id,
            'data_registo': str(data_registo),
            'peso': float(peso),
            'tomou_dose': tomou,
            'quantidade_dose': float(dose) if tomou else 0.0
        }, on_conflict='user_id,data_registo').execute()
    except Exception:
        st.error("Não foi possível salvar o registro no Supabase.")
        st.info(
            "Confirme que a tabela registos_diarios possui uma restrição única "
            "para user_id e data_registo e que as permissões authenticated estão ativas."
        )
        st.stop()

    st.session_state["ml_retreino_pendente"] = {
        "user_id": str(user_id),
        "data_registro": str(data_registo),
        "dose_registrada": bool(tomou),
    }


def exibir_status_retreino_ml(user_id, df_historico, treino_realizado):
    pendente = st.session_state.get("ml_retreino_pendente")
    if not pendente or pendente.get("user_id") != str(user_id):
        return

    if len(df_historico) <= 3:
        st.info(
            "Registro salvo. O retreino será feito quando este perfil tiver "
            "pelo menos quatro pesagens."
        )
        return

    if treino_realizado:
        data_registro = pd.to_datetime(pendente["data_registro"]).strftime('%d/%m/%Y')
        motivo = (
            "registro de dose e peso"
            if pendente["dose_registrada"]
            else "atualização da pesagem"
        )
        st.success(
            f"Modelo recalculado para este perfil após o {motivo} de {data_registro}, "
            f"usando {len(df_historico)} registros de peso."
        )
        st.session_state.pop("ml_retreino_pendente", None)


# --- 3. MOTOR DE INTELIGÊNCIA ARTIFICIAL ---
def criar_modelos_regressao():
    return {
        "Ridge": make_pipeline(
            PolynomialFeatures(degree=2),
            Ridge(alpha=10.0),
        ),
        "SVR": make_pipeline(
            StandardScaler(),
            SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1),
        ),
        "Huber": make_pipeline(
            StandardScaler(),
            HuberRegressor(),
        ),
    }


def gerar_predicao_ml(df_historico, peso_inicial):
    df_historico = df_historico.copy().sort_values('data_registo')
    df_historico['data_registo'] = pd.to_datetime(df_historico['data_registo'])
    data_inicio = df_historico['data_registo'].min()
    df_historico['Dias_Tratamento'] = (df_historico['data_registo'] - data_inicio).dt.days
    
    X = df_historico[['Dias_Tratamento']]
    y = df_historico['peso']

    modelos = criar_modelos_regressao()
    for modelo in modelos.values():
        modelo.fit(X, y)
    
    ultimo_dia = df_historico['Dias_Tratamento'].max()
    dias_alvo = np.array([10, 20, 30])
    dias_grafico = np.arange(ultimo_dia + 1, ultimo_dia + 31)
    datas_grafico = df_historico['data_registo'].max() + pd.to_timedelta(
        np.arange(1, 31), unit='D'
    )
    dias_grafico_df = pd.DataFrame({'Dias_Tratamento': dias_grafico})
    dias_alvo_df = pd.DataFrame({'Dias_Tratamento': ultimo_dia + dias_alvo})
    predicoes_grafico = {
        nome: modelo.predict(dias_grafico_df)
        for nome, modelo in modelos.items()
    }
    predicoes_alvo = {
        nome: modelo.predict(dias_alvo_df)
        for nome, modelo in modelos.items()
    }
    peso_atual = float(df_historico['peso'].iloc[-1])
    perda_atual = ((peso_inicial - peso_atual) / peso_inicial) * 100
    projecoes = {
        int(dias): {
            'peso': float(predicoes_alvo["Ridge"][indice]),
            'perda': float(
                ((peso_inicial - predicoes_alvo["Ridge"][indice]) / peso_inicial) * 100
            ),
            **{
                f'peso_{nome.lower()}': float(predicoes_alvo[nome][indice])
                for nome in ("SVR", "Huber")
            },
            **{
                f'perda_{nome.lower()}': float(
                    ((peso_inicial - predicoes_alvo[nome][indice]) / peso_inicial) * 100
                )
                for nome in ("SVR", "Huber")
            },
        }
        for indice, dias in enumerate(dias_alvo)
    }
    
    # Geração do Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_historico['data_registo'], y, color='black', label='Peso real', zorder=5)
    estilos_modelos = {
        "Ridge": ("blue", "Ridge polinomial"),
        "SVR": ("green", "SVR (RBF)"),
        "Huber": ("#d97706", "Huber (candidato)"),
    }
    datas_alvo = df_historico['data_registo'].max() + pd.to_timedelta(
        dias_alvo,
        unit='D',
    )
    for nome, modelo in modelos.items():
        cor, rotulo = estilos_modelos[nome]
        ax.plot(
            df_historico['data_registo'],
            modelo.predict(X),
            color=cor,
            linestyle='--',
            alpha=0.8,
            label=rotulo,
        )
        ax.plot(
            datas_grafico,
            predicoes_grafico[nome],
            color=cor,
            linestyle='--',
            linewidth=2,
        )
        ax.scatter(
            datas_alvo,
            predicoes_alvo[nome],
            color=cor,
            zorder=5,
            label=f'{nome}: pontos de 10, 20 e 30 dias',
        )
    
    ax.set_title("Evolução e projeção do peso")
    ax.set_ylabel("Peso (kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, peso_atual, perda_atual, projecoes


def classificar_variacao_peso(variacao_kg):
    if (
        variacao_kg < -LIMIAR_VARIACAO_PESO_KG
        and not np.isclose(
            variacao_kg,
            -LIMIAR_VARIACAO_PESO_KG,
            atol=1e-9,
            rtol=0.0,
        )
    ):
        return "Perda"
    if (
        variacao_kg > LIMIAR_VARIACAO_PESO_KG
        and not np.isclose(
            variacao_kg,
            LIMIAR_VARIACAO_PESO_KG,
            atol=1e-9,
            rtol=0.0,
        )
    ):
        return "Ganho"
    return "Estável"


def avaliar_regressores_ml(df_historico):
    if len(df_historico) < MIN_REGISTROS_AVALIACAO_ML:
        return None

    dados = df_historico.copy().sort_values('data_registo')
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    data_inicio = dados['data_registo'].iloc[0]
    dados['Dias_Tratamento'] = (dados['data_registo'] - data_inicio).dt.days
    X = dados[['Dias_Tratamento']]
    y = dados['peso'].astype(float)

    pesos_reais = []
    pesos_previstos = {nome: [] for nome in criar_modelos_regressao()}
    pesos_baseline = []
    for indice_teste in range(MIN_REGISTROS_TREINO_ML, len(dados)):
        pesos_reais.append(float(y.iloc[indice_teste]))
        pesos_baseline.append(float(y.iloc[indice_teste - 1]))
        for nome_modelo, modelo in criar_modelos_regressao().items():
            modelo.fit(X.iloc[:indice_teste], y.iloc[:indice_teste])
            pesos_previstos[nome_modelo].append(float(
                modelo.predict(X.iloc[[indice_teste]])[0]
            ))

    def calcular_erros(valores_previstos):
        return {
            "mae": float(mean_absolute_error(pesos_reais, valores_previstos)),
            "rmse": float(np.sqrt(mean_squared_error(pesos_reais, valores_previstos))),
        }

    return {
        "modelos": {
            nome: calcular_erros(valores)
            for nome, valores in pesos_previstos.items()
        },
        "baseline": calcular_erros(pesos_baseline),
        "quantidade_avaliacoes": len(pesos_reais),
    }


def avaliar_classificador_tendencia_ml(df_historico):
    if len(df_historico) < MIN_REGISTROS_AVALIACAO_ML:
        return None

    dados = df_historico.copy().sort_values('data_registo')
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    pesos = dados['peso'].astype(float).to_numpy()
    dias = (dados['data_registo'] - dados['data_registo'].iloc[0]).dt.days.to_numpy()
    datas = dados['data_registo'].to_numpy()

    linhas_features = []
    classes_reais = []
    for indice in range(1, len(dados)):
        variacoes_anteriores = np.diff(pesos[:indice])
        variacoes_recentes = variacoes_anteriores[-3:]
        intervalo_anterior = (
            float((datas[indice - 1] - datas[indice - 2]) / np.timedelta64(1, 'D'))
            if indice > 1
            else 0.0
        )
        linhas_features.append([
            float(dias[indice - 1]),
            float(variacoes_anteriores[-1]) if len(variacoes_anteriores) else 0.0,
            float(np.mean(variacoes_recentes)) if len(variacoes_recentes) else 0.0,
            intervalo_anterior,
        ])
        classes_reais.append(
            classificar_variacao_peso(pesos[indice] - pesos[indice - 1])
        )

    X = np.asarray(linhas_features, dtype=float)
    y = np.asarray(classes_reais, dtype=object)
    classes_previstas = []
    reais_avaliados = []
    scores_auc = []
    classes_auc = []
    classes_baseline = []

    for indice_teste in range(MIN_REGISTROS_TREINO_ML - 1, len(y)):
        classes_treino = y[:indice_teste]
        if len(np.unique(classes_treino)) < 2:
            continue

        modelo = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        modelo.fit(X[:indice_teste], classes_treino)
        classe_prevista = str(modelo.predict(X[[indice_teste]])[0])
        classes_previstas.append(classe_prevista)
        reais_avaliados.append(str(y[indice_teste]))
        classes_baseline.append(str(y[indice_teste - 1]))

        classificador = modelo.named_steps["logisticregression"]
        if all(classe in classificador.classes_ for classe in CLASSES_VARIACAO_PESO):
            probabilidades = modelo.predict_proba(X[[indice_teste]])[0]
            scores_auc.append([
                float(probabilidades[list(classificador.classes_).index(classe)])
                for classe in CLASSES_VARIACAO_PESO
            ])
            classes_auc.append(str(y[indice_teste]))

    if not reais_avaliados:
        return None

    matriz = pd.DataFrame(
        confusion_matrix(
            reais_avaliados,
            classes_previstas,
            labels=CLASSES_VARIACAO_PESO,
        ),
        index=CLASSES_VARIACAO_PESO,
        columns=CLASSES_VARIACAO_PESO,
    )
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        reais_avaliados,
        classes_previstas,
        labels=CLASSES_VARIACAO_PESO,
        zero_division=0,
    )
    metricas_por_classe = pd.DataFrame({
        "Classe": CLASSES_VARIACAO_PESO,
        "Precisão": precisao,
        "Recall": recall,
        "F1-score": f1,
        "Suporte": suporte,
    })
    metricas = {
        "acuracia": float(accuracy_score(reais_avaliados, classes_previstas)),
        "precisao_macro": float(np.mean(precisao)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "acuracia_baseline": float(accuracy_score(reais_avaliados, classes_baseline)),
        "auc_roc_macro_ovr": None,
        "quantidade_auc": len(classes_auc),
    }
    auc_classes_presentes = set(classes_auc)
    if (
        scores_auc
        and all(classe in auc_classes_presentes for classe in CLASSES_VARIACAO_PESO)
    ):
        matriz_auc = np.asarray(scores_auc)
        auc_por_classe = [
            roc_auc_score(
                np.asarray(classes_auc) == classe,
                matriz_auc[:, indice],
            )
            for indice, classe in enumerate(CLASSES_VARIACAO_PESO)
        ]
        metricas["auc_roc_macro_ovr"] = float(np.mean(auc_por_classe))

    return {
        "matriz": matriz,
        "metricas": metricas,
        "metricas_por_classe": metricas_por_classe,
        "quantidade_avaliacoes": len(reais_avaliados),
    }


def gerar_grafico_matriz_confusao(avaliacao):
    fig, eixo = plt.subplots(figsize=(6, 5))
    matriz = avaliacao["matriz"]
    eixo.imshow(matriz.to_numpy(), cmap='Blues')
    eixo.set_title(
        "LogisticRegression — matriz de confusão\n"
        f"Acurácia: {avaliacao['metricas']['acuracia']:.1%}"
    )
    eixo.set_xticks(range(len(CLASSES_VARIACAO_PESO)))
    eixo.set_xticklabels(CLASSES_VARIACAO_PESO)
    eixo.set_yticks(range(len(CLASSES_VARIACAO_PESO)))
    eixo.set_yticklabels(CLASSES_VARIACAO_PESO)
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
    for linha in range(len(CLASSES_VARIACAO_PESO)):
        for coluna in range(len(CLASSES_VARIACAO_PESO)):
            eixo.text(
                coluna,
                linha,
                str(int(matriz.iloc[linha, coluna])),
                ha="center",
                va="center",
                color="black",
            )
    fig.tight_layout()
    return fig


def gerar_grafico_metricas_regressao(avaliacao):
    linhas = [
        [
            "Huber (candidato)" if nome == "Huber" else nome,
            f"{metricas['mae']:.3f} kg",
            f"{metricas['rmse']:.3f} kg",
        ]
        for nome, metricas in avaliacao["modelos"].items()
    ]
    linhas.append([
        "Persistência (último peso)",
        f"{avaliacao['baseline']['mae']:.3f} kg",
        f"{avaliacao['baseline']['rmse']:.3f} kg",
    ])
    fig, eixo = plt.subplots(figsize=(9, 3.5))
    eixo.axis('off')
    eixo.set_title(
        f"Validação cronológica da regressão — "
        f"{avaliacao['quantidade_avaliacoes']} previsões"
    )
    tabela = eixo.table(
        cellText=linhas,
        colLabels=["Modelo", "MAE", "RMSE"],
        loc='center',
        cellLoc='center',
        bbox=[0.02, 0.08, 0.96, 0.78],
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    return fig


def criar_tabela_metricas_regressao(avaliacao):
    linhas = [
        {
            "Modelo": "Huber (candidato)" if nome == "Huber" else nome,
            "MAE (kg)": round(metricas["mae"], 3),
            "RMSE (kg)": round(metricas["rmse"], 3),
        }
        for nome, metricas in avaliacao["modelos"].items()
    ]
    linhas.append({
        "Modelo": "Persistência (último peso)",
        "MAE (kg)": round(avaliacao["baseline"]["mae"], 3),
        "RMSE (kg)": round(avaliacao["baseline"]["rmse"], 3),
    })
    return pd.DataFrame(linhas)


def criar_tabelas_metricas_classificacao(avaliacao):
    metricas = avaliacao["metricas"]
    auc = metricas["auc_roc_macro_ovr"]
    resumo = pd.DataFrame([{
        "Modelo": "LogisticRegression",
        "Acurácia": f"{metricas['acuracia']:.1%}",
        "Precisão macro": f"{metricas['precisao_macro']:.1%}",
        "Recall macro": f"{metricas['recall_macro']:.1%}",
        "F1-score macro": f"{metricas['f1_macro']:.1%}",
        "AUC-ROC macro OvR": f"{auc:.1%}" if auc is not None else "Indisponível",
        "Nº usados no AUC": metricas["quantidade_auc"],
        "Baseline persistente": f"{metricas['acuracia_baseline']:.1%}",
        "Nº validações": avaliacao["quantidade_avaliacoes"],
    }])
    por_classe = avaliacao["metricas_por_classe"].copy()
    for coluna in ("Precisão", "Recall", "F1-score"):
        por_classe[coluna] = por_classe[coluna].map(lambda valor: f"{valor:.1%}")
    return resumo, por_classe


def gerar_grafico_doses(df_historico):
    df_doses = df_historico.copy()
    df_doses['data_registo'] = pd.to_datetime(df_doses['data_registo'])
    df_doses = df_doses[df_doses['tomou_dose'] & (df_doses['quantidade_dose'] > 0)]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(df_doses['data_registo'], df_doses['quantidade_dose'], color='#2ca02c', width=0.8)
    ax.set_title("Doses registradas ao longo do tratamento")
    ax.set_ylabel("Dose (mg)")
    ax.set_xlabel("Data")
    ax.grid(axis='y', alpha=0.25)
    fig.autofmt_xdate()
    return fig


def filtrar_registros_semanais(df_historico):
    if df_historico.empty:
        return df_historico.copy()

    dados = df_historico.copy()
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    dados = dados.sort_values('data_registo')
    data_inicio = dados['data_registo'].iloc[0].normalize()
    dias_desde_inicio = (dados['data_registo'].dt.normalize() - data_inicio).dt.days
    registros_semanais = dias_desde_inicio.mod(7).eq(0)
    if dias_desde_inicio.iloc[-1] % 7:
        registros_semanais.iloc[-1] = True
    return dados.loc[registros_semanais].copy()


def filtrar_marcas_semanais_completas(df_semanal):
    if df_semanal.empty:
        return df_semanal.copy()

    dados = df_semanal.copy().sort_values('data_registo')
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    data_inicio = dados['data_registo'].iloc[0].normalize()
    dias_desde_inicio = (dados['data_registo'].dt.normalize() - data_inicio).dt.days
    return dados.loc[dias_desde_inicio.mod(7).eq(0)].copy()


def calcular_imc(peso, altura_m):
    return float(peso) / float(altura_m) ** 2


def gerar_grafico_semanal(
    df_semanal,
    semanas_projecao=0,
    df_semanal_completo=None,
    altura_m=None,
):
    datas = pd.to_datetime(df_semanal['data_registo'])
    pesos = df_semanal['peso']
    fig, ax = plt.subplots(figsize=(10, 5))
    valores = pesos if altura_m is None else pesos / float(altura_m) ** 2
    rotulo = "Peso (kg)" if altura_m is None else "IMC"
    nome_serie = "Peso registrado" if altura_m is None else "IMC registrado"
    ax.plot(datas, valores, marker='o', color='blue', label=nome_serie)
    for data, valor in zip(datas, valores):
        ax.annotate(
            f"{valor:.2f} kg" if altura_m is None else f"{valor:.1f}",
            (data, valor),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
        )

    datas_eixo = datas.tolist()
    if semanas_projecao > 0 and df_semanal_completo is not None:
        datas_ajuste = pd.to_datetime(
            df_semanal_completo['data_registo']
        ).reset_index(drop=True)
        pesos_ajuste = df_semanal_completo['peso'].to_numpy(dtype=float)
        dias_historicos = (datas_ajuste - datas_ajuste.iloc[0]).dt.days.to_numpy(dtype=float)
        inclinacao, intercepto = np.polyfit(
            dias_historicos,
            pesos_ajuste,
            1,
        )
        datas_futuras = pd.DatetimeIndex(
            [
                datas_ajuste.iloc[-1] + pd.Timedelta(days=7 * semana)
                for semana in range(1, semanas_projecao + 1)
            ]
        )
        dias_futuros_desde_inicio = (
            datas_futuras - datas_ajuste.iloc[0]
        ).days.to_numpy(dtype=float)
        pesos_futuros = inclinacao * dias_futuros_desde_inicio + intercepto
        valores_futuros = (
            pesos_futuros
            if altura_m is None
            else pesos_futuros / float(altura_m) ** 2
        )
        ax.plot(
            datas_futuras,
            valores_futuros,
            marker='o',
            linestyle='--',
            color='orange',
            label='Projeção linear',
        )
        for data, valor in zip(datas_futuras, valores_futuros):
            ax.annotate(
                f"{valor:.2f} kg" if altura_m is None else f"{valor:.1f}",
                (data, valor),
                xytext=(0, -16),
                textcoords='offset points',
                ha='center',
            )
        datas_eixo.extend(datas_futuras.tolist())

    datas_eixo = pd.DatetimeIndex(datas_eixo)
    ax.set_xticks(datas_eixo)
    ax.set_xticklabels(datas_eixo.strftime('%d/%m/%Y'), rotation=45, ha='right')
    titulo = "Histórico semanal do peso"
    if semanas_projecao > 0:
        titulo += f" com projeção linear de {semanas_projecao} semanas"
    if altura_m is not None:
        titulo = "Evolução semanal do IMC"
        if semanas_projecao > 0:
            titulo += f" com previsão linear de {semanas_projecao} semanas"
    ax.set_title(titulo)
    ax.set_ylabel(rotulo)
    ax.set_xlabel("Data")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def filtrar_historico(df_historico, chave="periodo_relatorio"):
    if df_historico.empty:
        return df_historico

    dados = df_historico.copy()
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    data_minima = dados['data_registo'].min().date()
    data_maxima = dados['data_registo'].max().date()
    filtro = st.selectbox(
        "Período",
        ["Todos os registros", "Últimos 7 dias", "Últimos 30 dias", "Últimos 90 dias", "Período personalizado"],
        key=chave,
    )

    if filtro == "Todos os registros":
        return dados
    if filtro == "Período personalizado":
        col_inicio, col_fim = st.columns(2)
        data_inicio = col_inicio.date_input("Data inicial", value=data_minima, min_value=data_minima, max_value=data_maxima, key=f"{chave}_inicio")
        data_fim = col_fim.date_input("Data final", value=data_maxima, min_value=data_minima, max_value=data_maxima, key=f"{chave}_fim")
        return dados[
            (dados['data_registo'].dt.date >= data_inicio)
            & (dados['data_registo'].dt.date <= data_fim)
        ] if data_inicio <= data_fim else dados.iloc[0:0]

    dias = {"Últimos 7 dias": 7, "Últimos 30 dias": 30, "Últimos 90 dias": 90}[filtro]
    data_inicio = max(data_minima, data_maxima - timedelta(days=dias - 1))
    return dados[dados['data_registo'].dt.date >= data_inicio]


def gerar_pdf_historico(
    df_historico,
    titulo,
    perfil=None,
    tipo='historico',
    peso_inicial=None,
    altura_m=None,
    df_historico_completo=None,
    avaliacao_regressoes=None,
    avaliacao_classificacao=None,
):
    dados = df_historico.copy()
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    historico_completo = (
        df_historico
        if df_historico_completo is None
        else df_historico_completo
    ).copy()
    historico_completo['data_registo'] = pd.to_datetime(
        historico_completo['data_registo']
    )
    tabela = dados[['data_registo', 'peso', 'tomou_dose', 'quantidade_dose']].copy()
    tabela.columns = ['Data', 'Peso (kg)', 'Tomou dose', 'Dose (mg)']
    tabela['Data'] = tabela['Data'].dt.strftime('%d/%m/%Y')
    tabela['Peso (kg)'] = tabela['Peso (kg)'].map(lambda valor: f'{valor:.1f}')
    tabela['Dose (mg)'] = tabela['Dose (mg)'].map(lambda valor: f'{valor:.2f}')

    def adicionar_marca(fig):
        if LOGO_PATH.exists():
            logo_ax = fig.add_axes([0.03, 0.02, 0.08, 0.05])
            logo_ax.imshow(plt.imread(LOGO_PATH))
            logo_ax.axis('off')
        fig.text(0.13, 0.035, 'Desenvolvido por Reinaldo Galvão', fontsize=8, color='#555555')

    def salvar_grafico_a4(pdf, figura):
        imagem = BytesIO()
        figura.savefig(imagem, format='png', dpi=150, bbox_inches='tight')
        imagem.seek(0)
        pagina, eixo = plt.subplots(figsize=(11.69, 8.27))
        eixo.set_position([0.06, 0.10, 0.88, 0.80])
        eixo.imshow(plt.imread(imagem), aspect='auto')
        eixo.axis('off')
        adicionar_marca(pagina)
        pdf.savefig(pagina)
        plt.close(pagina)
        plt.close(figura)

    arquivo = BytesIO()
    with PdfPages(arquivo) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        fig.text(0.08, 0.90, titulo, fontsize=20, fontweight='bold')
        fig.text(0.08, 0.85, f'Gerado em {date.today().strftime("%d/%m/%Y")}', fontsize=10)
        fig.text(0.08, 0.78, f'Registros: {len(dados)}', fontsize=12)
        fig.text(0.08, 0.74, f'Peso médio: {dados["peso"].mean():.1f} kg', fontsize=12)
        fig.text(0.08, 0.70, f'Dose total: {dados.loc[dados["tomou_dose"], "quantidade_dose"].sum():.2f} mg', fontsize=12)
        if perfil:
            nascimento = pd.to_datetime(perfil['data_nascimento']).strftime('%d/%m/%Y')
            fig.text(0.08, 0.62, f'Nome: {perfil["nome"]}', fontsize=11)
            fig.text(0.08, 0.58, f'Sexo: {perfil["sexo"]}', fontsize=11)
            fig.text(0.08, 0.54, f'Data de nascimento: {nascimento}', fontsize=11)
        adicionar_marca(fig)
        pdf.savefig(fig)
        plt.close(fig)

        if tipo == 'acompanhamento' and len(dados) > 3:
            figura, _, _, _ = gerar_predicao_ml(dados, peso_inicial)
            salvar_grafico_a4(pdf, figura)
        elif tipo == 'estatisticas':
            tabela_resumo = gerar_tabela_resumo_tratamento(
                historico_completo,
                perfil,
            )
            if altura_m is not None:
                dados_ordenados = historico_completo.sort_values('data_registo')
                altura_m = float(altura_m)
                tabela_imc = pd.DataFrame([
                    ("IMC inicial", f"{calcular_imc(peso_inicial, altura_m):.1f}"),
                    (
                        "IMC médio",
                        f"{calcular_imc(float(dados_ordenados['peso'].mean()), altura_m):.1f}",
                    ),
                    (
                        "IMC atual",
                        f"{calcular_imc(float(dados_ordenados['peso'].iloc[-1]), altura_m):.1f}",
                    ),
                ], columns=["Indicador", "Resultado"])
                tabela_resumo = pd.concat(
                    [tabela_resumo, tabela_imc],
                    ignore_index=True,
                )

            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            fig.text(
                0.08,
                0.93,
                'Resumo do tratamento e IMC',
                fontsize=18,
                fontweight='bold',
            )
            fig.text(
                0.08,
                0.89,
                'Dias contam da primeira à última pesagem, inclusive; registros de peso '
                'contam as pesagens nesse mesmo período.',
                fontsize=10,
            )
            tabela_pdf = ax.table(
                cellText=tabela_resumo.values,
                colLabels=tabela_resumo.columns,
                loc='center',
                cellLoc='left',
                colWidths=[0.62, 0.38],
                bbox=[0.05, 0.04, 0.90, 0.78],
            )
            tabela_pdf.auto_set_font_size(False)
            tabela_pdf.set_fontsize(9)
            tabela_pdf.scale(1, 1.2)
            salvar_grafico_a4(pdf, fig)

            tabela_variacoes = gerar_tabela_variacoes_peso(historico_completo)
            if tabela_variacoes.empty:
                figura_variacoes, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis('off')
                ax.text(
                    0.5,
                    0.55,
                    "São necessárias pelo menos duas pesagens para "
                    "classificar perda, ganho ou estabilidade.",
                    ha='center',
                    wrap=True,
                )
                salvar_grafico_a4(pdf, figura_variacoes)
            else:
                for inicio in range(0, len(tabela_variacoes), 25):
                    pagina_variacoes = tabela_variacoes.iloc[inicio:inicio + 25]
                    figura_variacoes, ax = plt.subplots(figsize=(11.69, 8.27))
                    ax.axis('off')
                    figura_variacoes.text(
                        0.08,
                        0.93,
                        "Datas e variações entre pesagens",
                        fontsize=18,
                        fontweight='bold',
                    )
                    figura_variacoes.text(
                        0.08,
                        0.89,
                        "Variação = peso registrado menos peso anterior; estável "
                        "entre -0,10 kg e +0,10 kg, inclusive.",
                        fontsize=10,
                    )
                    tabela_pdf = ax.table(
                        cellText=pagina_variacoes.values,
                        colLabels=pagina_variacoes.columns,
                        loc='center',
                        cellLoc='center',
                        colWidths=[0.20, 0.20, 0.20, 0.16, 0.18],
                        bbox=[0.04, 0.04, 0.92, 0.80],
                    )
                    tabela_pdf.auto_set_font_size(False)
                    tabela_pdf.set_fontsize(9)
                    salvar_grafico_a4(pdf, figura_variacoes)

            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.16)
            ax.plot(dados['data_registo'], dados['peso'], marker='o', color='#1f77b4', label='Peso registrado')
            if len(dados) >= 3:
                ax.plot(dados['data_registo'], dados['peso'].rolling(3, min_periods=1).mean(), color='#ff7f0e', linewidth=2, label='Média móvel (3 registros)')
            ax.set_title('Evolução do peso no período')
            ax.set_ylabel('Peso (kg)')
            ax.set_xlabel('Data')
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
            salvar_grafico_a4(pdf, fig)

            semanal = filtrar_registros_semanais(historico_completo)
            semanal_completo = filtrar_marcas_semanais_completas(semanal)
            semanas_projecao = 4 if len(semanal_completo) > 1 else 0
            figura_semanal = gerar_grafico_semanal(
                semanal,
                semanas_projecao=semanas_projecao,
                df_semanal_completo=semanal_completo,
            )
            salvar_grafico_a4(pdf, figura_semanal)

            if altura_m is not None:
                figura_imc_semanal = gerar_grafico_semanal(
                    semanal,
                    semanas_projecao=semanas_projecao,
                    df_semanal_completo=semanal_completo,
                    altura_m=altura_m,
                )
                salvar_grafico_a4(pdf, figura_imc_semanal)

            if len(historico_completo) > 3:
                figura_predicao, _, _, _ = gerar_predicao_ml(
                    historico_completo,
                    peso_inicial,
                )
                figura_predicao.text(
                    0.02,
                    0.01,
                    "Projeção estatística; não substitui orientação médica.",
                    fontsize=8,
                )
                salvar_grafico_a4(pdf, figura_predicao)

            dados_doses = dados[dados['tomou_dose'] & (dados['quantidade_dose'] > 0)]
            if not dados_doses.empty:
                figura_doses = gerar_grafico_doses(dados)
                salvar_grafico_a4(pdf, figura_doses)

            metricas_regressao = (
                avaliar_regressores_ml(historico_completo)
                if avaliacao_regressoes is None
                else avaliacao_regressoes
            )
            if metricas_regressao is not None:
                salvar_grafico_a4(
                    pdf,
                    gerar_grafico_metricas_regressao(metricas_regressao),
                )
            else:
                figura_regressao, ax = plt.subplots(figsize=(10, 5))
                ax.axis('off')
                ax.text(
                    0.5,
                    0.6,
                    "Métricas de regressão ainda indisponíveis",
                    ha='center',
                    fontsize=16,
                    fontweight='bold',
                )
                ax.text(
                    0.5,
                    0.4,
                    "São necessárias pelo menos seis pesagens para comparar "
                    "os modelos com validação cronológica.",
                    ha='center',
                    wrap=True,
                )
                salvar_grafico_a4(pdf, figura_regressao)

            avaliacao_ml = (
                avaliar_classificador_tendencia_ml(historico_completo)
                if avaliacao_classificacao is None
                else avaliacao_classificacao
            )
            if avaliacao_ml is not None:
                resumo_ml, por_classe_ml = criar_tabelas_metricas_classificacao(
                    avaliacao_ml
                )
                figura_metricas, eixos = plt.subplots(
                    2,
                    1,
                    figsize=(11.69, 8.27),
                    gridspec_kw={"height_ratios": [1, 1.2]},
                )
                figura_metricas.suptitle(
                    "Métricas de classificação da tendência",
                    fontsize=18,
                    fontweight='bold',
                )
                eixos[0].axis('off')
                linhas_resumo = [
                    (nome, valor)
                    for nome, valor in resumo_ml.iloc[0].items()
                ]
                eixos[0].table(
                    cellText=linhas_resumo,
                    colLabels=["Métrica", "Resultado"],
                    loc='center',
                    cellLoc='center',
                    bbox=[0.15, 0.02, 0.70, 0.92],
                ).set_fontsize(10)
                eixos[1].axis('off')
                eixos[1].set_title("Métricas por classe", fontsize=12)
                eixos[1].table(
                    cellText=por_classe_ml.values,
                    colLabels=por_classe_ml.columns,
                    loc='center',
                    cellLoc='center',
                    bbox=[0.08, 0.10, 0.84, 0.78],
                ).set_fontsize(10)
                figura_metricas.text(
                    0.5,
                    0.015,
                    "AUC-ROC macro OvR usa previsões fora da amostra em que todas "
                    "as classes estavam disponíveis no treino.",
                    ha='center',
                    fontsize=8,
                )
                figura_metricas.tight_layout(rect=[0, 0.04, 1, 0.95])
                salvar_grafico_a4(pdf, figura_metricas)

                salvar_grafico_a4(
                    pdf,
                    gerar_grafico_matriz_confusao(avaliacao_ml),
                )
            else:
                figura_classificacao, ax = plt.subplots(figsize=(10, 5))
                ax.axis('off')
                ax.text(
                    0.5,
                    0.6,
                    "Avaliação de classificação ainda indisponível",
                    ha='center',
                    fontsize=16,
                    fontweight='bold',
                )
                ax.text(
                    0.5,
                    0.4,
                    "São necessárias pelo menos seis pesagens e variação "
                    "suficiente para treinar o classificador.",
                    ha='center',
                    wrap=True,
                )
                salvar_grafico_a4(pdf, figura_classificacao)
        else:
            for inicio in range(0, len(tabela), 25):
                pagina = tabela.iloc[inicio:inicio + 25]
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis('off')
                tabela_pdf = ax.table(
                    cellText=pagina.values,
                    colLabels=pagina.columns,
                    loc='center',
                    cellLoc='center',
                )
                tabela_pdf.auto_set_font_size(False)
                tabela_pdf.set_fontsize(10)
                tabela_pdf.scale(1, 1.6)
                adicionar_marca(fig)
                pdf.savefig(fig)
                plt.close(fig)

    return arquivo.getvalue()


def exibir_download_pdf(
    df_historico,
    titulo,
    perfil,
    tipo='historico',
    peso_inicial=None,
    altura_m=None,
    df_historico_completo=None,
    avaliacao_regressoes=None,
    avaliacao_classificacao=None,
):
    if not df_historico.empty:
        st.download_button(
            "Baixar relatório em PDF",
            gerar_pdf_historico(
                df_historico,
                titulo,
                perfil,
                tipo,
                peso_inicial,
                altura_m,
                df_historico_completo,
                avaliacao_regressoes,
                avaliacao_classificacao,
            ),
            file_name="relatorio_semaglutida.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def exibir_relatorio(df_historico, perfil):
    st.header("📄 Histórico")
    if df_historico.empty:
        st.info("Ainda não há registros para o período selecionado.")
        return

    dados_filtrados = filtrar_historico(df_historico)

    if dados_filtrados.empty:
        st.info("Não há registros no período selecionado.")
        return

    doses = dados_filtrados.loc[dados_filtrados['tomou_dose'], 'quantidade_dose'].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", len(dados_filtrados))
    col2.metric("Peso médio", f"{dados_filtrados['peso'].mean():.1f} kg")
    col3.metric("Dose total", f"{doses:.2f} mg")

    tabela = dados_filtrados[
        ['data_registo', 'peso', 'tomou_dose', 'quantidade_dose']
    ].rename(columns={
        'data_registo': 'Data',
        'peso': 'Peso (kg)',
        'tomou_dose': 'Tomou dose',
        'quantidade_dose': 'Dose (mg)',
    })
    tabela['Data'] = tabela['Data'].dt.strftime('%d/%m/%Y')
    st.dataframe(tabela, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar relatório em CSV",
        tabela.to_csv(index=False).encode('utf-8-sig'),
        file_name="relatorio_semaglutida.csv",
        mime="text/csv",
        use_container_width=True,
    )
    exibir_download_pdf(dados_filtrados, "Histórico de semaglutida", perfil, tipo='historico')


def gerar_tabela_resumo_tratamento(df_historico, perfil):
    if df_historico.empty:
        return pd.DataFrame(columns=["Indicador", "Resultado"])

    dados = df_historico.copy()
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    dados = dados.sort_values('data_registo').reset_index(drop=True)
    variacoes = dados['peso'].astype(float).diff().dropna()
    classes_variacao = variacoes.map(classificar_variacao_peso)

    data_inicio = dados['data_registo'].iloc[0]
    data_fim = dados['data_registo'].iloc[-1]
    dias_em_tratamento = max((data_fim.date() - data_inicio.date()).days + 1, 1)
    peso_inicial = float(perfil['peso_inicial'])
    peso_atual = float(dados['peso'].iloc[-1])
    perda_total = peso_inicial - peso_atual
    perda_media_diaria = perda_total / dias_em_tratamento
    variacao_percentual = perda_total / peso_inicial * 100

    def formatar_variacao(valor, unidade, casas_decimais=2):
        valor = round(valor, casas_decimais)
        if valor > 0:
            return f"{valor:.{casas_decimais}f} {unidade} de perda"
        if valor < 0:
            return f"{abs(valor):.{casas_decimais}f} {unidade} de ganho"
        return f"{0:.{casas_decimais}f} {unidade} (sem alteração)"

    linhas = [
        ("Data de início (primeira pesagem registrada)", data_inicio.strftime('%d/%m/%Y')),
        ("Data da última pesagem", data_fim.strftime('%d/%m/%Y')),
        ("Número de dias em tratamento (período registrado)", str(dias_em_tratamento)),
        ("Número de registros de peso no período do tratamento", str(len(dados))),
        ("Peso inicial do perfil", f"{peso_inicial:.2f} kg"),
        ("Peso da última pesagem", f"{peso_atual:.2f} kg"),
        ("Perda de peso total", formatar_variacao(perda_total, "kg")),
        ("Perda média por dia", formatar_variacao(perda_media_diaria, "kg/dia", 3)),
        ("Variação desde o peso inicial", formatar_variacao(variacao_percentual, "%", 1)),
        ("Intervalos com perda", str(int(classes_variacao.eq("Perda").sum()))),
        ("Intervalos com ganho", str(int(classes_variacao.eq("Ganho").sum()))),
        ("Intervalos com peso estável", str(int(classes_variacao.eq("Estável").sum()))),
        ("Dias com dose registrada", str(int(dados['tomou_dose'].sum()))),
        (
            "Dose total registrada",
            f"{dados.loc[dados['tomou_dose'], 'quantidade_dose'].sum():.2f} mg",
        ),
        ("Peso médio registrado", f"{dados['peso'].mean():.2f} kg"),
    ]
    return pd.DataFrame(linhas, columns=["Indicador", "Resultado"])


def gerar_tabela_variacoes_peso(df_historico):
    colunas = [
        "Data da pesagem",
        "Peso anterior (kg)",
        "Peso registrado (kg)",
        "Variação (kg)",
        "Resultado",
    ]
    if len(df_historico) < 2:
        return pd.DataFrame(columns=colunas)

    dados = df_historico.copy()
    dados['data_registo'] = pd.to_datetime(dados['data_registo'])
    dados = dados.sort_values('data_registo').reset_index(drop=True)
    pesos = dados['peso'].astype(float)
    variacoes = pesos.diff()
    pesos_anteriores = pesos.shift(1)
    mascara = variacoes.notna()
    tabela = pd.DataFrame({
        "Data da pesagem": dados.loc[mascara, 'data_registo'].dt.strftime('%d/%m/%Y'),
        "Peso anterior (kg)": pesos_anteriores.loc[mascara].map(
            lambda valor: round(float(valor), 2)
        ),
        "Peso registrado (kg)": pesos.loc[mascara].map(
            lambda valor: round(float(valor), 2)
        ),
        "Variação (kg)": variacoes.loc[mascara].map(
            lambda valor: f"{valor:+.3f}"
        ),
        "Resultado": variacoes.loc[mascara].map(classificar_variacao_peso),
    })
    return tabela.reset_index(drop=True)


def exibir_estatisticas(df_historico, perfil):
    st.header("📊 Estatísticas")
    if df_historico.empty:
        st.info("Ainda não há dados suficientes para calcular estatísticas.")
        return

    st.subheader("Resumo do tratamento")
    st.caption(
        "Resumo descritivo, não é um diagnóstico clínico. Os dias contam da primeira "
        "à última pesagem registrada, inclusive; o número de registros conta as pesagens "
        "nesse mesmo período. A variação compara cada pesagem à anterior: perda abaixo "
        "de -0,10 kg, estável entre -0,10 kg e +0,10 kg (inclusive), e ganho acima "
        "de +0,10 kg."
    )
    tabela_resumo = gerar_tabela_resumo_tratamento(df_historico, perfil)
    st.dataframe(tabela_resumo, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar resumo do tratamento em CSV",
        tabela_resumo.to_csv(index=False).encode('utf-8-sig'),
        file_name="resumo_tratamento.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.subheader("Datas e variações entre pesagens")
    st.caption(
        "Cada linha classifica a variação na data da pesagem registrada, "
        "comparada à pesagem anterior. A primeira pesagem é apenas a referência. "
        "Variações de -0,10 kg a +0,10 kg, inclusive, são estáveis."
    )
    tabela_variacoes = gerar_tabela_variacoes_peso(df_historico)
    if tabela_variacoes.empty:
        st.info("São necessárias pelo menos duas pesagens para mostrar perda, ganho ou estabilidade.")
    else:
        st.dataframe(tabela_variacoes, use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar datas e variações em CSV",
            tabela_variacoes.to_csv(index=False).encode('utf-8-sig'),
            file_name="variacoes_peso.csv",
            mime="text/csv",
            use_container_width=True,
        )

    peso_inicial = float(perfil['peso_inicial'])
    altura_m = perfil.get('altura_m')
    if altura_m is None and 'altura_m' not in perfil:
        st.warning(
            "Para habilitar o cálculo do IMC, atualize o banco executando "
            "`database/migrar_altura_imc.sql` no SQL Editor do Supabase."
        )
    elif altura_m is None:
        st.info("Informe sua altura para calcular o IMC.")
        with st.form("form_altura_imc"):
            altura_m = st.number_input(
                "Altura (m)",
                min_value=1.0,
                max_value=2.5,
                value=1.7,
                step=0.01,
                format="%.2f",
            )
            if st.form_submit_button("Salvar altura"):
                supabase.table('utilizadores').update({
                    'altura_m': float(altura_m),
                }).eq('id', perfil['id']).execute()
                st.success("Altura salva.")
                st.rerun()

    dados = filtrar_historico(df_historico, chave="periodo_estatisticas")
    if dados.empty:
        st.info("Não há registros no período selecionado.")
        return

    peso_atual = float(dados['peso'].iloc[-1])
    peso_minimo = float(dados['peso'].min())
    peso_maximo = float(dados['peso'].max())
    peso_medio = float(dados['peso'].mean())
    perda_periodo_kg = float(dados['peso'].iloc[0]) - peso_atual
    perda_periodo = ((float(dados['peso'].iloc[0]) - peso_atual) / float(dados['peso'].iloc[0])) * 100
    dias_com_dose = int(dados['tomou_dose'].sum())
    dose_media = float(dados.loc[dados['tomou_dose'], 'quantidade_dose'].mean()) if dias_com_dose else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", len(dados))
    col2.metric("Peso atual", f"{peso_atual:.1f} kg")
    col3.metric("Perda no período", f"{perda_periodo:.1f}%")
    col3.caption(f"{perda_periodo_kg:.1f} kg")
    col4, col5, col6 = st.columns(3)
    col4.metric("Média do período", f"{peso_medio:.1f} kg")
    col5.metric("Menor peso", f"{peso_minimo:.1f} kg")
    col6.metric("Maior peso", f"{peso_maximo:.1f} kg")
    st.caption(f"Dose média nos dias registrados: {dose_media:.2f} mg. Peso inicial do perfil: {peso_inicial:.1f} kg.")

    if altura_m is not None:
        altura_m = float(altura_m)
        historico_completo = df_historico.sort_values('data_registo')
        imc_inicial = calcular_imc(peso_inicial, altura_m)
        imc_medio = calcular_imc(
            float(historico_completo['peso'].mean()),
            altura_m,
        )
        imc_atual = calcular_imc(
            float(historico_completo['peso'].iloc[-1]),
            altura_m,
        )
        st.subheader("IMC")
        col_imc_inicial, col_imc_medio, col_imc_atual = st.columns(3)
        col_imc_inicial.metric("IMC inicial", f"{imc_inicial:.1f}")
        col_imc_medio.metric("IMC médio", f"{imc_medio:.1f}")
        col_imc_atual.metric("IMC atual", f"{imc_atual:.1f}")
        st.caption(
            "IMC inicial calculado pelo peso inicial do perfil; média e valor atual "
            "usam todo o histórico registrado. O IMC é um indicador geral e não "
            "substitui avaliação profissional."
        )

    st.subheader("Evolução do peso no período")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dados['data_registo'], dados['peso'], marker='o', color='#1f77b4', label='Peso registrado')
    if len(dados) >= 3:
        ax.plot(dados['data_registo'], dados['peso'].rolling(3, min_periods=1).mean(), color='#ff7f0e', linewidth=2, label='Média móvel (3 registros)')
    ax.set_ylabel("Peso (kg)")
    ax.set_xlabel("Data")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

    historico_semanal = filtrar_registros_semanais(df_historico)
    marcas_semanais_completas = filtrar_marcas_semanais_completas(historico_semanal)
    semanas_previsao = 4 if len(marcas_semanais_completas) > 1 else 0
    st.subheader("Evolução semanal do peso com previsão")
    st.caption("Este gráfico considera todo o histórico, independentemente do filtro de período acima.")
    st.pyplot(
        gerar_grafico_semanal(
            historico_semanal,
            semanas_projecao=semanas_previsao,
            df_semanal_completo=marcas_semanais_completas,
        ),
        clear_figure=True,
    )
    if semanas_previsao == 0:
        st.info("São necessárias pelo menos duas semanas completas para calcular a previsão linear.")

    if altura_m is not None:
        st.subheader("Evolução semanal do IMC com previsão")
        st.pyplot(
            gerar_grafico_semanal(
                historico_semanal,
                semanas_projecao=semanas_previsao,
                df_semanal_completo=marcas_semanais_completas,
                altura_m=altura_m,
            ),
            clear_figure=True,
        )
        st.caption(
            "A previsão prolonga linearmente a tendência semanal observada; "
            "não é uma previsão clínica nem considera mudanças futuras."
        )

    st.subheader("Análise preditiva e histórico")
    if len(df_historico) > 3:
        figura_predicao, _, _, _ = gerar_predicao_ml(
            df_historico,
            peso_inicial,
        )
        treino_realizado = True
        st.pyplot(figura_predicao, clear_figure=True)
        st.caption(
            "O treino é recalculado com as datas e os pesos de todo o histórico deste perfil. "
            "O registro de dose dispara a atualização, mas a dose não é variável do modelo. "
            "A projeção é estatística e não substitui orientação médica."
        )
    else:
        treino_realizado = False
        st.info(
            "São necessários pelo menos quatro registros de peso para exibir "
            "a análise preditiva."
        )
    exibir_status_retreino_ml(perfil['id'], df_historico, treino_realizado)

    st.subheader("Doses no período")
    dados_doses = dados[dados['tomou_dose'] & (dados['quantidade_dose'] > 0)]
    if dados_doses.empty:
        st.info("Não há doses registradas no período selecionado.")
    else:
        st.pyplot(gerar_grafico_doses(dados), clear_figure=True)

    st.subheader("Comparação dos modelos de previsão de peso")
    avaliacao_regressoes = avaliar_regressores_ml(df_historico)
    if avaliacao_regressoes is None:
        st.info(
            "São necessárias pelo menos seis pesagens para comparar os modelos "
            "com validação cronológica."
        )
    else:
        st.caption(
            f"MAE e RMSE em kg, calculados em {avaliacao_regressoes['quantidade_avaliacoes']} "
            "previsões futuras (walk-forward). Valores menores indicam menor erro. "
            "A linha de persistência usa a última pesagem como previsão. Huber é "
            "avaliado como candidato e não substitui o SVR automaticamente."
        )
        st.dataframe(
            criar_tabela_metricas_regressao(avaliacao_regressoes),
            use_container_width=True,
            hide_index=True,
        )

    avaliacao_classificacao = avaliar_classificador_tendencia_ml(df_historico)
    st.subheader("Classificação da tendência — LogisticRegression")
    if avaliacao_classificacao is None:
        st.info(
            "A classificação aparecerá quando houver pelo menos seis pesagens "
            "e diversidade suficiente de tendências no histórico de treino."
        )
    else:
        resumo_classificacao, metricas_por_classe = (
            criar_tabelas_metricas_classificacao(avaliacao_classificacao)
        )
        st.caption(
            f"Validação cronológica com {avaliacao_classificacao['quantidade_avaliacoes']} "
            "previsões. Precisão, recall e F1 são médias macro e também são detalhados "
            "por classe. AUC-ROC é One-vs-Rest e só é exibida quando as três classes "
            "têm observações positivas e negativas avaliadas, após aparecerem no treino. "
            "A linha de baseline prevê a última tendência observada."
        )
        st.dataframe(
            resumo_classificacao,
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            metricas_por_classe,
            use_container_width=True,
            hide_index=True,
        )
        st.pyplot(
            gerar_grafico_matriz_confusao(avaliacao_classificacao),
            clear_figure=True,
        )

    exibir_download_pdf(
        dados,
        "Estatísticas do tratamento com semaglutida",
        perfil,
        tipo='estatisticas',
        peso_inicial=peso_inicial,
        altura_m=altura_m,
        df_historico_completo=df_historico,
        avaliacao_regressoes=avaliacao_regressoes,
        avaliacao_classificacao=avaliacao_classificacao,
    )

# --- 4. INTERFACE DO UTILIZADOR (FRONTEND) ---
st.title("📉 Acompanhamento com IA - Semaglutida")

user = get_authenticated_user()

if not user:
    st.info("Entre com a sua conta Google para continuar.")
    login_with_google()
    st.stop()

perfil = obter_perfil(user.id)

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=150)
    if perfil:
        nascimento_formatado = pd.to_datetime(perfil['data_nascimento']).strftime('%d/%m/%Y')
        st.markdown(f"**Nome:** {perfil['nome']}")
        st.markdown(f"**Sexo:** {perfil['sexo']}")
        st.markdown(f"**Data de nascimento:** {nascimento_formatado}")
    else:
        st.caption("Perfil ainda não preenchido")
    menu = st.radio("Menu", ["Acompanhamento", "Estatísticas", "Histórico"])
    st.caption("Desenvolvido por Reinaldo Galvão")
    if st.button("Sair", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state.pop("supabase_client", None)
        st.rerun()

if not perfil:
    st.warning("Complete seu perfil para começar.")
    metadata = user.user_metadata or {}
    nome_padrao = metadata.get("full_name") or metadata.get("name") or ""
    with st.form("onboarding"):
        nome = st.text_input("Nome completo", value=nome_padrao)
        nascimento = st.date_input("Data de nascimento", min_value=date(1940, 1, 1), max_value=date.today())
        sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])
        peso_ini = st.number_input("Peso inicial (kg)", min_value=30.0, max_value=250.0, step=0.05)
        altura_m = st.number_input(
            "Altura (m)",
            min_value=1.0,
            max_value=2.5,
            value=1.7,
            step=0.01,
            format="%.2f",
        )

        if st.form_submit_button("Criar Perfil"):
            supabase.table('utilizadores').insert({
                'id': user.id, 'email': user.email, 'nome': nome,
                'data_nascimento': str(nascimento), 'sexo': sexo,
                'peso_inicial': peso_ini, 'altura_m': altura_m,
            }).execute()
            st.success("Perfil criado com sucesso.")
            st.rerun()
else:
    st.success(f"Olá, {perfil['nome']}! Bem-vindo de volta.")

    df = obter_historico(perfil['id'])

    if menu == "Histórico":
        exibir_relatorio(df, perfil)
        st.stop()
    if menu == "Estatísticas":
        exibir_estatisticas(df, perfil)
        st.stop()

    # --- ZONA DE REGISTO DIÁRIO ---
    st.subheader("📝 Registrar peso")
    form_version = st.session_state.get("registro_form_version", 0)
    data_input = st.date_input("Data da medição", value=date.today(), key=f"data_registro_{form_version}")
    registro_do_dia = df[pd.to_datetime(df['data_registo']).dt.date == data_input] if not df.empty else df
    peso_padrao = float(registro_do_dia['peso'].iloc[0]) if not registro_do_dia.empty else (float(df['peso'].iloc[-1]) if not df.empty else perfil['peso_inicial'])
    tomou_padrao = bool(registro_do_dia['tomou_dose'].iloc[0]) if not registro_do_dia.empty else False
    dose_padrao = float(registro_do_dia['quantidade_dose'].iloc[0]) if not registro_do_dia.empty else 0.25
    with st.form(f"registro_diario_{form_version}"):
        col1, col2 = st.columns(2)
        opcoes_dose = [0.25, 0.5, 1.0, 2.0, 2.4]
        peso_input = col1.number_input("Peso (kg)", min_value=30.0, max_value=250.0, step=0.05, value=peso_padrao, key=f"peso_input_{form_version}_{data_input}")

        tomou_remedio = col2.checkbox("Tomei a dose de semaglutida neste dia", value=tomou_padrao, key=f"tomou_remedio_{form_version}_{data_input}")
        dose_input = col2.selectbox("Dose aplicada (mg)", opcoes_dose, index=opcoes_dose.index(dose_padrao) if dose_padrao in opcoes_dose else 0, key=f"dose_input_{form_version}_{data_input}") if tomou_remedio else 0.0

        col_salvar, col_cancelar = st.columns(2)
        salvar = col_salvar.form_submit_button("Salvar registro")
        cancelar = col_cancelar.form_submit_button("Cancelar")

        if cancelar:
            st.session_state.registro_form_version = form_version + 1
            st.rerun()
        if salvar:
            guardar_registo(perfil['id'], data_input, peso_input, tomou_remedio, dose_input)
            st.success("Registro salvo! A IA está recalculando sua curva...")
            st.rerun()

    if not df.empty:
        with st.expander("Ajustar registro existente"):
            datas_registradas = pd.to_datetime(df['data_registo']).dt.date.tolist()[::-1]
            data_ajuste = st.selectbox("Selecione a data para ajustar", datas_registradas)
            registro = df[pd.to_datetime(df['data_registo']).dt.date == data_ajuste].iloc[0]
            with st.form("ajuste_registro"):
                peso_ajuste = st.number_input(
                    "Peso registrado (kg)",
                    min_value=30.0,
                    max_value=250.0,
                    step=0.05,
                    value=float(registro['peso']),
                )
                tomou_ajuste = st.checkbox(
                    "Tomei a dose neste dia",
                    value=bool(registro['tomou_dose']),
                )
                dose_ajuste = st.selectbox(
                    "Dose aplicada (mg)",
                    [0.25, 0.5, 1.0, 2.0, 2.4],
                    index=[0.25, 0.5, 1.0, 2.0, 2.4].index(float(registro['quantidade_dose']))
                    if float(registro['quantidade_dose']) in [0.25, 0.5, 1.0, 2.0, 2.4]
                    else 0,
                ) if tomou_ajuste else 0.0
                if st.form_submit_button("Salvar ajuste"):
                    guardar_registo(perfil['id'], data_ajuste, peso_ajuste, tomou_ajuste, dose_ajuste)
                    st.success("Registro atualizado.")
                    st.rerun()

    # --- ZONA DA INTELIGÊNCIA ARTIFICIAL ---
    st.divider()
    st.subheader("🧠 Análise preditiva e histórico")

    treino_realizado = False
    if len(df) > 3:
        fig, peso_atual, perda_atual, projecoes = gerar_predicao_ml(
            df,
            perfil['peso_inicial'],
        )
        treino_realizado = True
        st.pyplot(fig)

        st.caption("Percentuais calculados em relação ao peso inicial informado no perfil.")
        st.caption("As projeções são estatísticas e não substituem orientação médica.")
        st.caption(
            "O treino é recalculado com as datas e os pesos deste perfil; doses "
            "registradas não são variáveis de entrada do modelo."
        )
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        col_met1.metric("Perda atual", f"{perda_atual:.1f}%", help=f"Peso atual: {peso_atual:.1f} kg")
        col_met1.caption(f"{perfil['peso_inicial'] - peso_atual:.1f} kg")
        col_met2.metric("Previsão em 10 dias", f"{projecoes[10]['perda']:.1f}%", help=f"Peso projetado: {projecoes[10]['peso']:.1f} kg")
        col_met2.caption(f"{perfil['peso_inicial'] - projecoes[10]['peso']:.1f} kg")
        col_met3.metric("Previsão em 20 dias", f"{projecoes[20]['perda']:.1f}%", help=f"Peso projetado: {projecoes[20]['peso']:.1f} kg")
        col_met3.caption(f"{perfil['peso_inicial'] - projecoes[20]['peso']:.1f} kg")
        col_met4.metric("Previsão em 30 dias", f"{projecoes[30]['perda']:.1f}%", help=f"Peso projetado: {projecoes[30]['peso']:.1f} kg")
        col_met4.caption(f"{perfil['peso_inicial'] - projecoes[30]['peso']:.1f} kg")

        st.subheader("💉 Histórico de doses")
        df_doses = df[df['tomou_dose'] & (df['quantidade_dose'] > 0)]
        if not df_doses.empty:
            st.pyplot(gerar_grafico_doses(df), clear_figure=True)
        else:
            st.info("Ainda não há doses registradas para exibir neste gráfico.")

        perda_30d = projecoes[30]['perda']
        if perda_30d > 15:
            st.info("🔵 Ritmo projetado acelerado (acima da referência clínica)")
        elif perda_30d >= 6:
            st.success("🟢 Ritmo projetado esperado (de acordo com a referência clínica)")
        else:
            st.warning("🟠 Ritmo projetado lento (abaixo da referência clínica)")
    else:
        st.info("Continue registrando seu peso por mais alguns dias para a IA calcular uma curva personalizada.")

    exibir_status_retreino_ml(perfil['id'], df, treino_realizado)

    st.divider()
    st.caption("Relatório completo")
    exibir_download_pdf(df, "Acompanhamento de semaglutida", perfil, tipo='acompanhamento', peso_inicial=perfil['peso_inicial'])