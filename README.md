# Semaglutida

Aplicação Streamlit para acompanhar peso e registos de dose de semaglutida, com uma projeção estatística baseada no histórico do utilizador.

## Requisitos

- Python 3.10 ou superior
- Projeto Supabase configurado

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edite `.streamlit/secrets.toml` com os valores do seu projeto Supabase e a URL pública da aplicação. Esse arquivo é ignorado pelo Git e não deve ser commitado.

## Google e Supabase Auth

1. No Google Cloud, crie um OAuth Client ID do tipo Web application.
2. No Supabase, ative o provider Google em **Authentication > Providers > Google** e informe o Client ID e o Client Secret.
3. No Google Cloud, use `https://SEU_PROJETO.supabase.co/auth/v1/callback` como redirect URI autorizado.
4. No Supabase, configure a Site URL e adicione o valor de `APP_URL` às Redirect URLs.
5. Execute `database/schema.sql` no SQL Editor do Supabase antes de iniciar a aplicação.

O `APP_URL` deve ser exatamente a URL acessível pelo navegador, incluindo o protocolo e a porta. Em produção, use HTTPS.

## Execução

```powershell
streamlit run semaglutide.py
```

A aplicação abre por padrão em `http://localhost:8501`.

## Estrutura

- `semaglutide.py`: entrada e interface do app
- `requirements.txt`: dependências Python
- `.streamlit/secrets.toml.example`: modelo de configuração local
- `database/schema.sql`: tabelas e políticas RLS necessárias no Supabase
- `Semaglutida.ipynb`: notebook original

> Esta aplicação é um protótipo de acompanhamento e não substitui avaliação ou orientação médica.
