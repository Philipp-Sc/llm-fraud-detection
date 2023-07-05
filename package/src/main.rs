use std::thread;
use std::time::Duration;

use rust_bert_fraud_detection_tools::service::spawn_rust_bert_fraud_detection_socket_service;
use std::env;
use rust_bert_fraud_detection_socket_ipc::ipc::client_send_rust_bert_fraud_detection_request;
use rust_bert_fraud_detection_tools::build::classification::feature_importance;
use rust_bert_fraud_detection_tools::build::create_naive_bayes_model;
use rust_bert_fraud_detection_tools::build::data::{generate_shuffled_idx, split_vector};

pub const SENTENCES: [&str;6] = [
    "You!!! Lose up to 19% weight. Special promotion on our new weightloss.",
    "Hi Bob, can you send me your machine learning homework?",
    "Don't forget our special promotion: -30% on men shoes, only today!",
    "Hi Bob, don't forget our meeting today at 4pm.",
    "⚠️ FINAL: LAST TERRA PHOENIX AIRDROP 🌎 ✅ CLAIM NOW All participants in this vote will receive a reward..",
    "Social KYC oracle (TYC)  PFC is asking for 20k Luna to build a social KYC protocol.."
    ];


// To just test the fraud detection:
//      sudo docker run -it --rm -v "$(pwd)/rustbert_cache":/usr/rustbert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release

// Start service container:
//      sudo docker run -d --rm -v "$(pwd)/rustbert_cache":/usr/rustbert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release start_service
//      (To later stop the service container)
//          sudo docker container ls
//          sudo docker stop CONTAINER_ID
// Run service test:
//      sudo docker run -it --rm -v "$(pwd)/rustbert_cache":/usr/rustbert_cache -v "$(pwd)/target":/usr/target -v "$(pwd)/cargo_home":/usr/cargo_home -v "$(pwd)/package":/usr/workspace -v "$(pwd)/tmp":/usr/workspace/tmp -v "$(pwd)/socket_ipc":/usr/socket_ipc rust-bert-fraud-detection cargo run --release test_service

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        println!("No command specified.");
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "naive_bayes_train_and_train_and_test_final_regression_model" => {naive_bayes_train_and_train_and_test_final_regression_model();},
        "naive_bayes_train" => {naive_bayes_train();},
        "naive_bayes_predict" => {naive_bayes_predict();},
        "train_and_test_final_regression_model_eval" => {train_and_test_final_regression_model(true);},
        "train_and_test_final_regression_model" => {train_and_test_final_regression_model(false);},
        "train_and_test_final_nn_model" => {train_and_test_final_nn_model(false);},
        "generate_feature_vectors" => {generate_feature_vectors();},
        "service" => {service();},
        "feature_selection" => {feature_selection();},
        _ => {panic!()}
    }

    Ok(())
}

fn service() -> anyhow::Result<()> {

    let mut args: Vec<String> = env::args().collect();
    args.reverse();
    args.pop();
    args.reverse();
    println!("env::args().collect(): {:?}",args);

    if args.len() <= 1 {
        println!("{:?}", &SENTENCES);
        let fraud_probabilities: Vec<f64> = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
        println!("Predictions:\n{:?}", fraud_probabilities);
        println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
        Ok(())
    }else{
        match args[1].as_str() {
            "start" => {
                spawn_rust_bert_fraud_detection_socket_service("./tmp/rust_bert_fraud_detection_socket").join().unwrap();
                Ok(())
            },
            "test" => {
                let result = client_send_rust_bert_fraud_detection_request("./tmp/rust_bert_fraud_detection_socket",SENTENCES.iter().map(|x|x.to_string()).collect::<Vec<String>>())?;
                println!("{:?}",result);
                Ok(())
            }
            _ => {
                println!("invalid command");
                Ok(())
            }
        }
    }
}

fn naive_bayes_train_and_train_and_test_final_regression_model() -> anyhow::Result<()> {
    let data_paths = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("data_gen_v5_({}).json",x)).collect::<Vec<String>>();
    let paths = data_paths.iter().map(|x| x.as_str()).collect::<Vec<&str>>();

    let shuffled_idx = generate_shuffled_idx(&paths[..])?;
   
    let data_paths1 = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("./dataset/{}.csv",x)).collect::<Vec<String>>();
    let paths1 = data_paths1.iter().map(|x| x.as_str()).collect::<Vec<&str>>();

    let dataset = rust_bert_fraud_detection_tools::build::data::read_datasets_and_shuffle(&paths1[..],&shuffled_idx)?;

    let (train_dataset, test_dataset) = split_vector(&dataset,0.8);
    let train_dataset = train_dataset.to_vec();
    let test_dataset = test_dataset.to_vec();

    create_naive_bayes_model(&train_dataset,&test_dataset)?;

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&paths[..],&shuffled_idx)?;

    let (x_train, x_test) = split_vector(&x_dataset,0.8);
    let x_train = x_train.to_vec();
    let x_test = x_test.to_vec();
    let (y_train, y_test) = split_vector(&y_dataset,0.8);
    let y_train = y_train.to_vec();
    let y_test = y_test.to_vec();

    rust_bert_fraud_detection_tools::build::create_classification_model(&x_train,&y_train)?;
    rust_bert_fraud_detection_tools::build::test_classification_model(&x_test,&y_test)?;

    let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())

}


fn feature_selection() -> anyhow::Result<()> {
    let data_paths = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("data_gen_v5_({}).json", x)).collect::<Vec<String>>();
    let paths = data_paths.iter().map(|x| x.as_str()).collect::<Vec<&str>>();

    let shuffled_idx = generate_shuffled_idx(&paths[..])?;

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&paths[..], &shuffled_idx)?;


    feature_importance(&x_dataset, &y_dataset)
}


fn train_and_test_final_nn_model(eval: bool) -> anyhow::Result<()> {

    let data_paths = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("data_gen_v5_({}).json",x)).collect::<Vec<String>>();
    let paths = data_paths.iter().map(|x| x.as_str()).collect::<Vec<&str>>();

    let shuffled_idx = generate_shuffled_idx(&paths[..])?;

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&paths[..],&shuffled_idx)?;

    // drop all features given a blacklist
/*
    let importance = vec![0.005900003154049609, 0.0041208350312930825, 0.0025368131416679575, 0.004062583993382742, 0.0032750838679589415, 0.008652334252391272, 0.00593174291330368, 0.0052959331514525745, 0.009203107625109145, 0.003761572040470753, 0.003582037602108143, 0.00780039060797062, 0.003648775640035148, 0.004434497715496505, 0.011206718624537433, 0.00620723435062582, 0.006574665965352202, 0.0034062290258876976, 0.032794123803081186, 0.008919884064939588, 0.0043351557111339775, 0.005520235111562328, 0.0041454345576169135, 0.0029296602801467246, 0.04691123690753006, 0.0027452620054188082, 0.005607814264524495, 0.005605517558308827, 0.009441739168879882, 0.006098148152458866, 0.0050199295901746386, 0.003291777818153207, 0.005136715957287582, 0.006891429696486162, 0.002149048614579839, 0.003936855805035756, 0.006069529936422121, 0.005353846079879107, 0.0021346815832244707, 0.008152627377781867, 0.006494339382107207, 0.005529995006374414, 0.004376178854223897, 0.0027242491671540126, 0.0038651498618062934, 0.008252657689690787, 0.005945655154831591, 0.00501647880215852, 0.003948668099984683, 0.006135331676477557, 0.009855971019667752, 0.0068390398568557165, 0.016545836740637376, 0.004647659801763336, 0.004002348420646261, 0.007550812361950647, 0.011290940730435826, 0.003330715898929376, 0.004948222164420129, 0.005343354854262659, 0.012233116362463613, 0.0036788194474375933, 0.006751604860173664, 0.003028977370104937, 0.006520171314983333, 0.010974105554464568, 0.004063459660578166, 0.004959340554830744, 0.003627294897551213, 0.007110788834306301, 0.004910377485531306, 0.0038863278411230707, 0.0040010204187505664, 0.004449257866485185, 0.0031996807415375864, 0.015173522147256528, 0.010224874977351257, 0.006018264079169756, 0.00825715081116186, 0.022803363170303653, 0.007201269737152977, 0.00781125756543272, 0.004372542972972421, 0.01583651705972147, 0.004244900902647618, 0.003972591326447519, 0.008593705776838894, 0.003121624175695368, 0.004840984617851394, 0.014901577055234589, 0.00476231448146218, 0.007268878528345079, 0.004241787459283724, 0.01580566606971197, 0.03684411074134846, 0.004342677835456931, 0.004357170070351489, 0.005157893841208267, 0.0037179393600163787, 0.0038184544386535885, 0.005219756386570415, 0.0056628196479023485, 0.027940337311708037, 0.003338120061659352, 0.015286470962016557, 0.003756716885789879, 0.019524210156056012, 0.008589196032440542, 0.036669126452287315, 0.029013546987538587, 0.02835563892888449, 0.006626778619521814, 0.003830775141126419, 0.01339794966015276, 0.013634060715143496, 0.004230287886605023, 0.0067884797940927066, 0.0033969295342272782, 0.005080058567560264, 0.0048183141304684405, 0.005928341400320177, 0.005550211167470146, 0.010488547420518548, 0.003638880713737473, 0.004555268921449786, 0.00515222196067851, 0.00789319213354385, 0.004503556170508907, 0.009896674270623181, 0.004069283492425447, 0.004616188781552608, 0.011075279541378541, 0.00691800815041087, 0.003972668799735755, 0.004270788117071738, 0.0058802499839991215, 0.008909378605584647, 0.002641242118547244, 0.0039865610718007585, 0.006255987619450784, 0.009334293646339602, 0.005496590967174411, 0.0048787167880381295, 0.003818502148469188, 0.005485801191711753, 0.00770079593499535, 0.007124230666339587, 0.0035936122402011055, 0.013032876544674223, 0.003484481915269962, 0.006397415250293256, 0.0022094030574071406, 0.009633795411105681, 0.005100705910938554, 0.007472701271588124, 0.006168444424225833, 0.14208074954988376, 0.3985863423201774, 0.029406976115361624, 0.05072013818401902, 0.032868717073914214, 1.891019666739516e-5, 0.013845833801094155, 0.0058732928891172854, 0.02994326629473548, 0.045243339951067274, 0.03093559093435697, 0.009271590473926101];

    let x_dataset = x_dataset.iter().map(|features| {
        let mut features_new = Vec::new();
        for i in 0..importance.len(){
            if importance[i]>=0.01 {
                features_new.push(features[i]);
            }
        }
        features_new
    }).collect();
*/

    if !eval {
        rust_bert_fraud_detection_tools::build::classification::deep_learning::train_nn(&x_dataset,&y_dataset);
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

        rust_bert_fraud_detection_tools::build::classification::deep_learning::train_nn(&x_train,&y_train);
        // rust_bert_fraud_detection_tools::build::test_classification_model(&x_test, &y_test)?;
    }
   // let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
   // println!("Predictions:\n{:?}",fraud_probabilities);
   // println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}

fn train_and_test_final_regression_model(eval: bool) -> anyhow::Result<()> {

    let data_paths = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("data_gen_v5_({}).json",x)).collect::<Vec<String>>();
    let paths = data_paths.iter().map(|x| x.as_str()).collect::<Vec<&str>>();

    let shuffled_idx = generate_shuffled_idx(&paths[..])?;

    let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&paths[..],&shuffled_idx)?;

    // drop all features given a blacklist

    let importance = vec![0.005900003154049609, 0.0041208350312930825, 0.0025368131416679575, 0.004062583993382742, 0.0032750838679589415, 0.008652334252391272, 0.00593174291330368, 0.0052959331514525745, 0.009203107625109145, 0.003761572040470753, 0.003582037602108143, 0.00780039060797062, 0.003648775640035148, 0.004434497715496505, 0.011206718624537433, 0.00620723435062582, 0.006574665965352202, 0.0034062290258876976, 0.032794123803081186, 0.008919884064939588, 0.0043351557111339775, 0.005520235111562328, 0.0041454345576169135, 0.0029296602801467246, 0.04691123690753006, 0.0027452620054188082, 0.005607814264524495, 0.005605517558308827, 0.009441739168879882, 0.006098148152458866, 0.0050199295901746386, 0.003291777818153207, 0.005136715957287582, 0.006891429696486162, 0.002149048614579839, 0.003936855805035756, 0.006069529936422121, 0.005353846079879107, 0.0021346815832244707, 0.008152627377781867, 0.006494339382107207, 0.005529995006374414, 0.004376178854223897, 0.0027242491671540126, 0.0038651498618062934, 0.008252657689690787, 0.005945655154831591, 0.00501647880215852, 0.003948668099984683, 0.006135331676477557, 0.009855971019667752, 0.0068390398568557165, 0.016545836740637376, 0.004647659801763336, 0.004002348420646261, 0.007550812361950647, 0.011290940730435826, 0.003330715898929376, 0.004948222164420129, 0.005343354854262659, 0.012233116362463613, 0.0036788194474375933, 0.006751604860173664, 0.003028977370104937, 0.006520171314983333, 0.010974105554464568, 0.004063459660578166, 0.004959340554830744, 0.003627294897551213, 0.007110788834306301, 0.004910377485531306, 0.0038863278411230707, 0.0040010204187505664, 0.004449257866485185, 0.0031996807415375864, 0.015173522147256528, 0.010224874977351257, 0.006018264079169756, 0.00825715081116186, 0.022803363170303653, 0.007201269737152977, 0.00781125756543272, 0.004372542972972421, 0.01583651705972147, 0.004244900902647618, 0.003972591326447519, 0.008593705776838894, 0.003121624175695368, 0.004840984617851394, 0.014901577055234589, 0.00476231448146218, 0.007268878528345079, 0.004241787459283724, 0.01580566606971197, 0.03684411074134846, 0.004342677835456931, 0.004357170070351489, 0.005157893841208267, 0.0037179393600163787, 0.0038184544386535885, 0.005219756386570415, 0.0056628196479023485, 0.027940337311708037, 0.003338120061659352, 0.015286470962016557, 0.003756716885789879, 0.019524210156056012, 0.008589196032440542, 0.036669126452287315, 0.029013546987538587, 0.02835563892888449, 0.006626778619521814, 0.003830775141126419, 0.01339794966015276, 0.013634060715143496, 0.004230287886605023, 0.0067884797940927066, 0.0033969295342272782, 0.005080058567560264, 0.0048183141304684405, 0.005928341400320177, 0.005550211167470146, 0.010488547420518548, 0.003638880713737473, 0.004555268921449786, 0.00515222196067851, 0.00789319213354385, 0.004503556170508907, 0.009896674270623181, 0.004069283492425447, 0.004616188781552608, 0.011075279541378541, 0.00691800815041087, 0.003972668799735755, 0.004270788117071738, 0.0058802499839991215, 0.008909378605584647, 0.002641242118547244, 0.0039865610718007585, 0.006255987619450784, 0.009334293646339602, 0.005496590967174411, 0.0048787167880381295, 0.003818502148469188, 0.005485801191711753, 0.00770079593499535, 0.007124230666339587, 0.0035936122402011055, 0.013032876544674223, 0.003484481915269962, 0.006397415250293256, 0.0022094030574071406, 0.009633795411105681, 0.005100705910938554, 0.007472701271588124, 0.006168444424225833, 0.14208074954988376, 0.3985863423201774, 0.029406976115361624, 0.05072013818401902, 0.032868717073914214, 1.891019666739516e-5, 0.013845833801094155, 0.0058732928891172854, 0.02994326629473548, 0.045243339951067274, 0.03093559093435697, 0.009271590473926101];

    let x_dataset = x_dataset.iter().map(|features| {
        let mut features_new = Vec::new();
        for i in 0..importance.len(){
            if importance[i]>=0.01 {
                features_new.push(features[i]);
            }
        }
        features_new
    }).collect();


    if !eval {
         rust_bert_fraud_detection_tools::build::create_classification_model(&x_dataset,&y_dataset)?;
         rust_bert_fraud_detection_tools::build::test_classification_model(&x_dataset,&y_dataset)?;
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

        rust_bert_fraud_detection_tools::build::create_classification_model(&x_train, &y_train)?;
        rust_bert_fraud_detection_tools::build::test_classification_model(&x_test, &y_test)?;
    }
    let fraud_probabilities = rust_bert_fraud_detection_tools::fraud_probabilities(&SENTENCES)?;
    println!("Predictions:\n{:?}",fraud_probabilities);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}


fn naive_bayes_predict() -> anyhow::Result<()>{
    let predictions = rust_bert_fraud_detection_tools::build::naive_bayes::categorical_nb_model_predict(SENTENCES.iter().map(|&s| s.to_string()).collect::<Vec<String>>())?;
    println!("Predictions:\n{:?}",predictions);
    println!("Labels:\n[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]");
    Ok(())
}


fn naive_bayes_train() -> anyhow::Result<()>{

    let data_paths = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("data_gen_v5_({}).json",x)).collect::<Vec<String>>();
    let paths = data_paths.iter().map(|x| x.as_str()).collect::<Vec<&str>>();

    let shuffled_idx = generate_shuffled_idx(&paths[..])?;

    let data_paths = vec![
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"].into_iter().map(|x| format!("./dataset/{}.csv",x)).collect::<Vec<String>>();
    let paths = data_paths.iter().map(|x| x.as_str()).collect::<Vec<&str>>();


    let dataset = rust_bert_fraud_detection_tools::build::data::read_datasets_and_shuffle(&paths[..],&shuffled_idx)?;

/*  let (train_dataset, test_dataset) = split_vector(&dataset,0.7);
    let train_dataset = train_dataset.to_vec();
    let test_dataset = test_dataset.to_vec();

    create_naive_bayes_model(&train_dataset,&test_dataset)
*/
    create_naive_bayes_model(&dataset,&dataset)
}

fn generate_feature_vectors() -> anyhow::Result<()> {

    // test governance spam ham
    // let training_data_path = "new_data_gen_v5_(governance_proposal_spam_likelihood).json";
    // rust_bert_fraud_detection_tools::build::create_training_data(vec!["./dataset/governance_proposal_spam_likelihood.csv"],training_data_path)?;


    let datasets = [
        "youtubeSpamCollection",
        "enronSpamSubset",
        "lingSpam",
        "smsspamcollection",
        "completeSpamAssassin",
        "governance_proposal_spam_likelihood"];

    for dataset in datasets {
        let training_data_path = format!("data_gen_v5_({}).json",dataset);
        let dataset_path = format!("./dataset/{}.csv",dataset);

        rust_bert_fraud_detection_tools::build::create_training_data(vec![&dataset_path],&training_data_path)?;
    }


    return Ok(());

}
