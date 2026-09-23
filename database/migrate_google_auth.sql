begin;

create temporary table migration_users (
    user_id uuid primary key,
    email text not null unique
) on commit drop;

insert into migration_users (user_id, email) values
    ('5a9eb68e-0f72-413c-bb0f-9e2e696eda6f', 'glaucianaffonseca@gmail.com'),
    ('2480f018-822c-4bda-bfce-25e075876eac', 'reinaldogalvao@gmail.com');

-- Stop before changing data if either Google account is not in Supabase Auth.
do $$
begin
    if exists (
        select 1
        from migration_users expected
        left join auth.users auth_user on auth_user.id = expected.user_id
        where auth_user.id is null
           or auth_user.email is null
           or lower(auth_user.email) <> lower(expected.email)
    ) then
        raise exception 'Os UUIDs/e-mails não correspondem aos usuários em auth.users.';
    end if;
end
$$;

-- The old schema references public.utilizadores.id. Remove those foreign keys
-- temporarily so the profile UUIDs can be replaced with auth.users UUIDs.
do $$
declare
    foreign_key record;
begin
    for foreign_key in
        select conname
        from pg_constraint
        where conrelid = 'public.registos_diarios'::regclass
          and contype = 'f'
    loop
        execute format(
            'alter table public.registos_diarios drop constraint %I',
            foreign_key.conname
        );
    end loop;
end
$$;

-- Abort instead of silently orphaning profiles that were not mapped.
do $$
begin
    if exists (
        select 1
        from public.utilizadores profile
        left join migration_users expected
            on lower(expected.email) = lower(profile.email)
        where expected.user_id is null
    ) then
        raise exception 'Existem perfis sem mapeamento Google em public.utilizadores.';
    end if;
end
$$;

-- Point existing daily records at the new authenticated UUIDs.
update public.registos_diarios record
set user_id = expected.user_id
from public.utilizadores profile
join migration_users expected
    on lower(expected.email) = lower(profile.email)
where record.user_id = profile.id;

-- Point profiles at auth.users and normalize the e-mail from the mapping.
update public.utilizadores profile
set id = expected.user_id,
    email = expected.email
from migration_users expected
where lower(expected.email) = lower(profile.email);

alter table public.utilizadores alter column id drop default;

-- Restore the production foreign keys.
do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conrelid = 'public.utilizadores'::regclass
          and conname = 'utilizadores_id_auth_users_fkey'
    ) then
        alter table public.utilizadores
            add constraint utilizadores_id_auth_users_fkey
            foreign key (id) references auth.users(id) on delete cascade;
    end if;

    if not exists (
        select 1 from pg_constraint
        where conrelid = 'public.registos_diarios'::regclass
          and conname = 'registos_diarios_user_id_fkey'
    ) then
        alter table public.registos_diarios
            add constraint registos_diarios_user_id_fkey
            foreign key (user_id) references public.utilizadores(id) on delete cascade;
    end if;
end
$$;

alter table public.utilizadores enable row level security;
alter table public.registos_diarios enable row level security;

drop policy if exists "demo leitura utilizadores" on public.utilizadores;
drop policy if exists "demo criação utilizadores" on public.utilizadores;
drop policy if exists "demo leitura registos" on public.registos_diarios;
drop policy if exists "demo criação registos" on public.registos_diarios;
drop policy if exists "demo atualização registos" on public.registos_diarios;
drop policy if exists "utilizadores_select_own" on public.utilizadores;
drop policy if exists "utilizadores_insert_own" on public.utilizadores;
drop policy if exists "utilizadores_update_own" on public.utilizadores;
drop policy if exists "registos_select_own" on public.registos_diarios;
drop policy if exists "registos_insert_own" on public.registos_diarios;
drop policy if exists "registos_update_own" on public.registos_diarios;
drop policy if exists "registos_delete_own" on public.registos_diarios;

create policy "utilizadores_select_own"
    on public.utilizadores for select
    to authenticated
    using (auth.uid() = id);

create policy "utilizadores_insert_own"
    on public.utilizadores for insert
    to authenticated
    with check (auth.uid() = id and lower(auth.jwt() ->> 'email') = lower(email));

create policy "utilizadores_update_own"
    on public.utilizadores for update
    to authenticated
    using (auth.uid() = id)
    with check (auth.uid() = id);

create policy "registos_select_own"
    on public.registos_diarios for select
    to authenticated
    using (auth.uid() = user_id);

create policy "registos_insert_own"
    on public.registos_diarios for insert
    to authenticated
    with check (auth.uid() = user_id);

create policy "registos_update_own"
    on public.registos_diarios for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "registos_delete_own"
    on public.registos_diarios for delete
    to authenticated
    using (auth.uid() = user_id);

commit;
