create table if not exists public.utilizadores (
    id uuid primary key references auth.users(id) on delete cascade,
    email text not null unique,
    nome text not null,
    data_nascimento date not null,
    sexo text not null check (sexo in ('Feminino', 'Masculino')),
    peso_inicial numeric(5, 2) not null check (peso_inicial between 30 and 250),
    created_at timestamptz not null default now()
);

create table if not exists public.registos_diarios (
    id uuid primary key default gen_random_uuid(),
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

grant usage on schema public to authenticated;
grant select, insert, update on public.utilizadores to authenticated;
grant select, insert, update, delete on public.registos_diarios to authenticated;

alter table public.utilizadores enable row level security;
alter table public.registos_diarios enable row level security;

drop policy if exists "utilizadores_select_own" on public.utilizadores;
create policy "utilizadores_select_own"
    on public.utilizadores for select
    to authenticated
    using (auth.uid() = id);

drop policy if exists "utilizadores_insert_own" on public.utilizadores;
create policy "utilizadores_insert_own"
    on public.utilizadores for insert
    to authenticated
    with check (auth.uid() = id and lower(auth.jwt() ->> 'email') = lower(email));

drop policy if exists "utilizadores_update_own" on public.utilizadores;
create policy "utilizadores_update_own"
    on public.utilizadores for update
    to authenticated
    using (auth.uid() = id)
    with check (auth.uid() = id);

drop policy if exists "registos_select_own" on public.registos_diarios;
create policy "registos_select_own"
    on public.registos_diarios for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists "registos_insert_own" on public.registos_diarios;
create policy "registos_insert_own"
    on public.registos_diarios for insert
    to authenticated
    with check (auth.uid() = user_id);

drop policy if exists "registos_update_own" on public.registos_diarios;
create policy "registos_update_own"
    on public.registos_diarios for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "registos_delete_own" on public.registos_diarios;
create policy "registos_delete_own"
    on public.registos_diarios for delete
    to authenticated
    using (auth.uid() = user_id);