create extension if not exists "uuid-ossp";

create table if not exists public.utilizadores (
    id uuid primary key default uuid_generate_v4(),
    email text not null unique,
    nome text not null,
    data_nascimento date not null,
    sexo text not null check (sexo in ('Feminino', 'Masculino')),
    peso_inicial numeric(5, 2) not null check (peso_inicial between 30 and 250),
    created_at timestamptz not null default now()
);

create table if not exists public.registos_diarios (
    id uuid primary key default uuid_generate_v4(),
    user_id uuid not null references public.utilizadores(id) on delete cascade,
    data_registo date not null,
    peso numeric(5, 2) not null check (peso between 30 and 250),
    tomou_dose boolean not null default false,
    quantidade_dose numeric(4, 2) not null default 0 check (quantidade_dose >= 0),
    created_at timestamptz not null default now(),
    unique (user_id, data_registo)
);

create index if not exists registos_diarios_user_data_idx
    on public.registos_diarios (user_id, data_registo);

alter table public.utilizadores enable row level security;
alter table public.registos_diarios enable row level security;

-- O app atual usa email como identificação de demonstração.
-- Substitua estas políticas por políticas baseadas em auth.uid() antes de produção.
create policy "demo leitura utilizadores"
    on public.utilizadores for select
    using (true);

create policy "demo criação utilizadores"
    on public.utilizadores for insert
    with check (true);

create policy "demo leitura registos"
    on public.registos_diarios for select
    using (true);

create policy "demo criação registos"
    on public.registos_diarios for insert
    with check (true);

create policy "demo atualização registos"
    on public.registos_diarios for update
    using (true)
    with check (true);
