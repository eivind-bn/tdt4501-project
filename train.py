
import xai

model = xai.V7Bot("cpu")

pop = model.populate(number_of_genomes=600)

pop.evolve(
    number_of_generations=600,
    survivor_cnt=1,
    elite_parents=0,
    roulette_parents=60,
    dirname="v7bot",
    number_of_process=96
)


