-- AutoLabeler platform schema for production Postgres.
-- Local development uses backend/data/platform.db via sqlite with the same domain model.

create table if not exists workspaces (
  id text primary key,
  name text not null,
  description text,
  created_at timestamptz not null default now()
);

create table if not exists projects (
  id text primary key,
  workspace_id text not null references workspaces(id),
  name text not null,
  description text,
  status text not null default 'active',
  created_at timestamptz not null default now()
);

create table if not exists dataset_versions (
  id text primary key,
  dataset_id text not null,
  project_id text references projects(id),
  version_name text not null,
  source text not null,
  manifest_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists lineage_events (
  id text primary key,
  dataset_id text not null,
  version_id text references dataset_versions(id),
  event_type text not null,
  inputs_json jsonb not null default '{}'::jsonb,
  outputs_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists audit_events (
  id text primary key,
  action text not null,
  resource_type text not null,
  resource_id text not null,
  metadata_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists worker_queue (
  id text primary key,
  task_type text not null,
  payload_json jsonb not null default '{}'::jsonb,
  status text not null default 'queued',
  attempts integer not null default 0,
  locked_by text,
  locked_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists model_runs (
  id text primary key,
  model_name text not null,
  task text not null,
  inputs_json jsonb not null default '{}'::jsonb,
  outputs_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists observability_metrics (
  id text primary key,
  name text not null,
  value double precision not null,
  labels_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists export_validations (
  id text primary key,
  job_id text not null,
  valid boolean not null,
  issues_json jsonb not null default '[]'::jsonb,
  statistics_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists preference_items (
  id text primary key,
  project_id text references projects(id),
  image_id text not null,
  prompt text not null,
  candidates_json jsonb not null default '[]'::jsonb,
  status text not null default 'open',
  created_at timestamptz not null default now()
);

create table if not exists preference_votes (
  id text primary key,
  item_id text not null references preference_items(id),
  selected_candidate_id text not null,
  rationale text,
  created_at timestamptz not null default now()
);

create table if not exists evaluation_reports (
  id text primary key,
  job_id text not null,
  report_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists quality_scores (
  id text primary key,
  job_id text not null,
  image_id text not null,
  score double precision not null,
  issues_json jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_projects_workspace on projects(workspace_id);
create index if not exists idx_dataset_versions_dataset on dataset_versions(dataset_id);
create index if not exists idx_lineage_dataset on lineage_events(dataset_id);
create index if not exists idx_audit_resource on audit_events(resource_type, resource_id);
create index if not exists idx_worker_queue_status on worker_queue(status, created_at);
create index if not exists idx_model_runs_created on model_runs(created_at desc);
create index if not exists idx_metrics_created on observability_metrics(created_at desc);
create index if not exists idx_preference_items_project on preference_items(project_id);
create index if not exists idx_evaluation_reports_job on evaluation_reports(job_id, created_at desc);
create index if not exists idx_quality_scores_job on quality_scores(job_id, score);
