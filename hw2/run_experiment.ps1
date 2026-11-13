# run.ps1
Write-Host "Lancement des experiences..." -ForegroundColor Cyan

uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
foreach ($lambda in 0, 0.95, 0.98, 0.99, 1) {
    uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda $lambda --exp_name "lunar_lander_lambda$lambda"
}


Write-Host "Toutes les experiences sont terminees !" -ForegroundColor Green
Read-Host "Appuyez sur Entrée pour quitter..."