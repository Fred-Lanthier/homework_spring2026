# run.ps1
Write-Host "Lancement des experiences..." -ForegroundColor Cyan

uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml
uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum_autotune.yaml
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune.yaml
uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_singleq.yaml
uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_clipq.yaml


Write-Host "Toutes les experiences sont terminees !" -ForegroundColor Green
Read-Host "Appuyez sur Entrée pour quitter..."