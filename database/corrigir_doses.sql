begin;

-- Prévia dos registros que serão normalizados.
select
    r.id,
    r.user_id,
    u.email,
    r.data_registo,
    r.quantidade_dose,
    case
        when r.quantidade_dose = 2.5 then 0.25
        when r.quantidade_dose = 5.0 then 0.5
    end::numeric as nova_dose
from public.registos_diarios r
join public.utilizadores u on u.id = r.user_id
where r.user_id in (
    '5a9eb68e-0f72-413c-bb0f-9e2e696eda6f'::uuid,
    '2480f018-822c-4bda-bfce-25e075876eac'::uuid
)
and r.quantidade_dose in (2.5, 5.0);

-- Os valores antigos estavam 10x maiores que a dose real registrada.
update public.registos_diarios
set quantidade_dose = case
    when quantidade_dose = 2.5 then 0.25
    when quantidade_dose = 5.0 then 0.5
end,
    tomou_dose = true
where user_id in (
    '5a9eb68e-0f72-413c-bb0f-9e2e696eda6f'::uuid,
    '2480f018-822c-4bda-bfce-25e075876eac'::uuid
)
and quantidade_dose in (2.5, 5.0);

-- Impede novos valores fora do menu do aplicativo.
do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conrelid = 'public.registos_diarios'::regclass
          and conname = 'registos_diarios_quantidade_dose_valida'
    ) then
        alter table public.registos_diarios
            add constraint registos_diarios_quantidade_dose_valida
            check (quantidade_dose in (0, 0.25, 0.5, 1.0, 2.0, 2.4));
    end if;
end
$$;

grant select, insert, update, delete
on public.registos_diarios
to authenticated;

notify pgrst, 'reload schema';

commit;
