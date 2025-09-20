from tqdm import tqdm
from pykeen.models import ERModel
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.predict import predict_target

def custom_link_prediction(model,training_triples_factory : TriplesFactory, evaluation_triples_factory: TriplesFactory, testing_triples_factory : TriplesFactory):
        testing_triples = testing_triples_factory.mapped_triples
        triples_to_filter = set(map(tuple, training_triples_factory.mapped_triples.tolist() + evaluation_triples_factory.mapped_triples.tolist()))
        metricas = {f"hits_at_{rank}" : 0.0 for rank in [1,3,5,10]}

        for triple in tqdm(testing_triples, desc="Evaluando link prediction: "):
            h_id = triple[0].tolist()
            r_id = triple[1].tolist()
            t_id = triple[2].tolist()

            predictions = predict_target(model=model, head=h_id, tail=t_id, triples_factory=training_triples_factory).df            
            
            filtered_predictions = predictions[predictions["relation_id"].apply(
                lambda r_id : not (h_id, r_id, t_id) in triples_to_filter
            )]

            filtered_prediction_list = filtered_predictions["relation_id"].to_list()[:10]
            
            for rank in [1,3,5,10]:
                key = f"hits_at_{rank}"
                metricas[key] = metricas[key] + (1.0 if r_id in filtered_prediction_list[:rank] else 0.0)
                
        for k in metricas:
            metricas[k] = metricas[k]/len(testing_triples)

        return metricas