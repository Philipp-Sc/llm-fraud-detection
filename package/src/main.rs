use std::thread;
use std::time::Duration;

use rust_bert_fraud_detection_tools::service::spawn_rust_bert_fraud_detection_socket_service;
use std::env;
use rust_bert_fraud_detection_socket_ipc::ipc::client_send_rust_bert_fraud_detection_request;
use rust_bert_fraud_detection_tools::build::classification::feature_importance;
use rust_bert_fraud_detection_tools::build::create_naive_bayes_model;
use rust_bert_fraud_detection_tools::build::data::{generate_shuffled_idx, split_vector};
use rust_bert_fraud_detection_tools::build::classification::deep_learning::{feature_importance_nn, get_new_nn, Predictor, z_score_normalize};
use rust_bert_fraud_detection_tools::build::language_model::load_embeddings_from_file;

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
        "train_and_test_final_nn_model_eval" => {train_and_test_final_nn_model(true);},
        "generate_feature_vectors" => {generate_feature_vectors();},
        "service" => {service();},
        "feature_selection" => {feature_selection();},
        "feature_selection_nn" => {feature_selection_nn();},

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

fn feature_selection_nn() -> anyhow::Result<()> {
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
    let (x_dataset, mean, std_dev) = z_score_normalize(&x_dataset, None);


    feature_importance_nn(&x_dataset, &y_dataset)
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

    //let shuffled_idx = generate_shuffled_idx(&paths[..])?;
    //let (x_dataset, y_dataset) = rust_bert_fraud_detection_tools::build::data::create_dataset(&paths[..],&shuffled_idx)?;

    let (x_dataset, y_dataset) =  load_embeddings_from_file(&paths[..])?;


    let (x_dataset, mean, std_dev) = z_score_normalize(&x_dataset, None);
    // drop all features given a blacklist

    //let importance = vec![0.005900003154049609, 0.0041208350312930825, 0.0025368131416679575, 0.004062583993382742, 0.0032750838679589415, 0.008652334252391272, 0.00593174291330368, 0.0052959331514525745, 0.009203107625109145, 0.003761572040470753, 0.003582037602108143, 0.00780039060797062, 0.003648775640035148, 0.004434497715496505, 0.011206718624537433, 0.00620723435062582, 0.006574665965352202, 0.0034062290258876976, 0.032794123803081186, 0.008919884064939588, 0.0043351557111339775, 0.005520235111562328, 0.0041454345576169135, 0.0029296602801467246, 0.04691123690753006, 0.0027452620054188082, 0.005607814264524495, 0.005605517558308827, 0.009441739168879882, 0.006098148152458866, 0.0050199295901746386, 0.003291777818153207, 0.005136715957287582, 0.006891429696486162, 0.002149048614579839, 0.003936855805035756, 0.006069529936422121, 0.005353846079879107, 0.0021346815832244707, 0.008152627377781867, 0.006494339382107207, 0.005529995006374414, 0.004376178854223897, 0.0027242491671540126, 0.0038651498618062934, 0.008252657689690787, 0.005945655154831591, 0.00501647880215852, 0.003948668099984683, 0.006135331676477557, 0.009855971019667752, 0.0068390398568557165, 0.016545836740637376, 0.004647659801763336, 0.004002348420646261, 0.007550812361950647, 0.011290940730435826, 0.003330715898929376, 0.004948222164420129, 0.005343354854262659, 0.012233116362463613, 0.0036788194474375933, 0.006751604860173664, 0.003028977370104937, 0.006520171314983333, 0.010974105554464568, 0.004063459660578166, 0.004959340554830744, 0.003627294897551213, 0.007110788834306301, 0.004910377485531306, 0.0038863278411230707, 0.0040010204187505664, 0.004449257866485185, 0.0031996807415375864, 0.015173522147256528, 0.010224874977351257, 0.006018264079169756, 0.00825715081116186, 0.022803363170303653, 0.007201269737152977, 0.00781125756543272, 0.004372542972972421, 0.01583651705972147, 0.004244900902647618, 0.003972591326447519, 0.008593705776838894, 0.003121624175695368, 0.004840984617851394, 0.014901577055234589, 0.00476231448146218, 0.007268878528345079, 0.004241787459283724, 0.01580566606971197, 0.03684411074134846, 0.004342677835456931, 0.004357170070351489, 0.005157893841208267, 0.0037179393600163787, 0.0038184544386535885, 0.005219756386570415, 0.0056628196479023485, 0.027940337311708037, 0.003338120061659352, 0.015286470962016557, 0.003756716885789879, 0.019524210156056012, 0.008589196032440542, 0.036669126452287315, 0.029013546987538587, 0.02835563892888449, 0.006626778619521814, 0.003830775141126419, 0.01339794966015276, 0.013634060715143496, 0.004230287886605023, 0.0067884797940927066, 0.0033969295342272782, 0.005080058567560264, 0.0048183141304684405, 0.005928341400320177, 0.005550211167470146, 0.010488547420518548, 0.003638880713737473, 0.004555268921449786, 0.00515222196067851, 0.00789319213354385, 0.004503556170508907, 0.009896674270623181, 0.004069283492425447, 0.004616188781552608, 0.011075279541378541, 0.00691800815041087, 0.003972668799735755, 0.004270788117071738, 0.0058802499839991215, 0.008909378605584647, 0.002641242118547244, 0.0039865610718007585, 0.006255987619450784, 0.009334293646339602, 0.005496590967174411, 0.0048787167880381295, 0.003818502148469188, 0.005485801191711753, 0.00770079593499535, 0.007124230666339587, 0.0035936122402011055, 0.013032876544674223, 0.003484481915269962, 0.006397415250293256, 0.0022094030574071406, 0.009633795411105681, 0.005100705910938554, 0.007472701271588124, 0.006168444424225833, 0.14208074954988376, 0.3985863423201774, 0.029406976115361624, 0.05072013818401902, 0.032868717073914214, 1.891019666739516e-5, 0.013845833801094155, 0.0058732928891172854, 0.02994326629473548, 0.045243339951067274, 0.03093559093435697, 0.009271590473926101];
    let importance = vec![-0.0007271265613121732, -0.002461960572321517, -0.0029777686391221853, -0.0021628862113937317, -0.0004584757003227088, 0.002003587608864468, -0.0012597322968002209, -0.0006439746612857719, -0.0016606487706311674, -0.001022674692931536, -0.0028752174739748696, -0.002016655677614621, 0.0006839873627388839, 0.0015387448899967786, -0.0029800715260467417, -0.002709051270846445, -0.0017848967337883816, -0.0005973865804406899, -0.003052897293667044, -0.0020789427049134153, -0.0011170173871435552, -0.002484941418739433, -0.002361528401857502, -0.0022316127164343376, -0.0009500209127765645, -0.00254703384005693, -0.0014414150071440591, 0.0015623207718911142, 0.0005993366143142416, -0.0033211647527588294, -0.0007260723182968141, -0.0017870690714165314, -0.0039309157495572185, -0.0009871708911597383, 0.0012522916046087658, 0.00175399448396316, 0.0007648165960952813, -0.00045519314656814716, -0.0032078341776469272, -0.001366462211555081, 0.0005325845186646771, -0.0023821643899492153, -0.004434628017552507, -0.0021045341698316426, -0.002932284022155956, 0.00058505373757773, -0.0005088712281116072, -0.0010923083329237918, -0.001591803387825806, 0.0008225358156937793, -0.002329687469075071, -0.0014404190346014938, -0.0038682212336651565, -0.002292359126397054, 6.445449996616886e-5, -0.003676133674551291, -0.003217771134984625, -0.0030233602129437986, -0.0028637946605894068, -0.000304410798545122, 0.000248984406987899, -0.002777908717683557, 0.00010477851460340506, -0.004241154888436995, -0.0011030848810873782, -0.0032920779361037228, -0.000993934703744007, 0.0002691825465740432, -0.00023682204156962554, -0.0009337028050981039, -0.003231591362883554, -0.001423338371607959, -0.0008059973499602977, -0.0005578038658525735, -0.002654211015499906, 0.0021212281359540893, -0.001537841749473254, -0.0018055602253894932, -0.0013645967986523442, 0.0016012898670921358, -0.0008304821335382096, -0.0017708739853378689, -0.0001340180074646167, 0.0012815600468218888, -0.003937062533251019, -0.0013058109126665705, -0.002881733366844347, -0.0022497862082344992, -0.0009100709769875491, 0.0016177047420429327, -0.0017803753217779648, 0.001734869470883296, 0.0032001925961566346, 6.867164394566289e-5, -0.0016573289807152692, -0.0009871492157983756, -0.0034354416121640478, -0.0014274626416065433, -0.0014649616082246855, 0.00014748611172512931, 0.0008390043157950213, 0.00011552253466169865, -0.0018020078760918534, -0.00186548160546034, -0.004057542869770665, -0.0030852023840780775, 0.0005189025851709565, -0.001190823719796866, -0.0017429297935792638, -0.003908694060388375, -0.0011715794683323237, -0.003334755962397562, -0.0020806983398486962, -0.00019391319906173298, -0.00276056099330239, -0.0020718188708967957, -0.0025623554874637665, -0.003378590080625773, -0.001979121841844746, 0.0021740665068745906, -0.004348232343951392, -0.0006337531120154252, -0.00277022983720211, -0.00266917085111581, -0.0039908443277236845, -0.0038427474824036093, -0.0003629499815133843, -0.002317601393654468, -0.0027311282314159166, -0.004897365699672568, -0.0024374645287481805, -0.004193285731752414, -0.005722895521087102, -0.0047148986617428406, -0.0018562397086072978, 0.0015112761602960587, -0.002840490586892606, -0.00301847759744117, -0.00020516118102379118, -0.0011578789249338881, -0.003961744325374933, 0.0005531065535091424, -0.0034492257573904062, -0.0010121169607148186, 0.0003632775920238796, 0.0010973642642216287, -0.0004416248054982854, 5.0867282917726253e-5, -0.000495453919746772, -0.002445996626211839, -0.0033514495612418827, 0.0012062879916322426, -0.003611121719262063, -0.00023975580996972402, -0.002741483122009041, -0.0009576603202347473, -0.004079053395411901, -0.0036559322179858596, -0.005519789237770509, 0.00019806237196154687, -0.002320278615421279, 0.0014569352057761003, -0.002887716750198108, 0.00040150020843988065, -0.004629734686027312, 0.0017952848700521081, 0.0021559522001455603, -0.0025841206739588133];

    let x_dataset = x_dataset.iter().map(|features| {
        let mut features_new = Vec::new();
        for i in 0..importance.len(){
            if importance[i]>=0.00010477851460340506 {
                features_new.push(features[i]);
            }
        }
        features_new
    }).collect();


    if !eval {
        let nn = rust_bert_fraud_detection_tools::build::classification::deep_learning::train_nn(&x_dataset,&y_dataset);
        let path = std::path::Path::new("./NeuralNet.bin");
        nn.save(path).unwrap();
        let mut nn = get_new_nn(x_dataset[0].len() as i64);
        nn.load(path).unwrap();
        rust_bert_fraud_detection_tools::build::classification::deep_learning::test_nn(&nn,&x_dataset,&y_dataset);
    }else {
        let (x_train, x_test) = split_vector(&x_dataset, 0.8);
        let x_train = x_train.to_vec();
        let x_test = x_test.to_vec();
        let (y_train, y_test) = split_vector(&y_dataset, 0.8);
        let y_train = y_train.to_vec();
        let y_test = y_test.to_vec();

        let nn = rust_bert_fraud_detection_tools::build::classification::deep_learning::train_nn(&x_train,&y_train);
        rust_bert_fraud_detection_tools::build::classification::deep_learning::test_nn(&&nn,&x_train,&y_train);
        rust_bert_fraud_detection_tools::build::classification::deep_learning::test_nn(&&nn,&x_test,&y_test);
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
