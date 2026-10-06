alter table public.utilizadores
    add column if not exists altura_m numeric(3, 2);

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conrelid = 'public.utilizadores'::regclass
          and conname = 'utilizadores_altura_m_check'
    ) then
        alter table public.utilizadores
            add constraint utilizadores_altura_m_check
            check (altura_m is null or altura_m between 1.00 and 2.50);
    end if;
end
$$;
