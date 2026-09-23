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

Edite `.streamlit/secrets.toml` com os valores do seu projeto Supabase. Esse arquivo é ignorado pelo Git e não deve ser commitado.

## Base de dados

Execute `supabase/schema.sql` no SQL Editor do Supabase antes de iniciar a aplicação.

## Execução

```powershell
streamlit run semaglutide.py
```

A aplicação abre por padrão em `http://localhost:8501`.

## Estrutura

- `semaglutide.py`: entrada e interface do app
- `requirements.txt`: dependências Python
- `.streamlit/secrets.toml.example`: modelo de configuração local
- `supabase/schema.sql`: tabelas necessárias no Supabase
- `Semaglutida.ipynb`: notebook original

> Esta aplicação é um protótipo de acompanhamento e não substitui avaliação ou orientação médica.
