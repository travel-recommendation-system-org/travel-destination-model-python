# run_all.py
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("EXECUTION COMPLETE DU SYSTEME DE RECOMMANDATION")
print("=" * 70)

# Étape 1: Exécuter main.py
print("\nÉTAPE 1: Exécution de main.py...")
from main import main
system_components = main()

# Étape 2: Exécuter les tests
print("\nÉTAPE 2: Exécution des tests...")
from test import run_all_tests
run_all_tests(system_components)

print("\n" + "=" * 70)
print("EXECUTION TERMINEE AVEC SUCCES")
print("=" * 70)