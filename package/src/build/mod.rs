use std::fs::File;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub mod data;
pub mod classification;
pub mod language_model;
pub mod feature_engineering;
pub mod sentiment;
pub mod naive_bayes;
pub mod feature_selection;


use data::read_datasets;


pub const OLD_FRAUD_INDICATORS: [&str;156] = [
    "Reliable, trustworthy references or sources",
    "Neutral language, no leading statements",
    "Educational content",
    "Promotion of inclusivity and diversity",
    "Censorship or freedom of speech issues",
    "Healthy debate and exchange of ideas",
    "Appropriate content rating for audience",
    "No personal attacks or defamatory language",
    "Promotion of mental health awareness",
    "Ethical sourcing or supply chain",
    "Public interest or societal impact",
    "Use of data and privacy considerations",
    "Intentional disinformation or propaganda",
    "Promotion of literacy and critical thinking",
    "Disclosure of conflicts of interest", // 21
    "Culturally sensitive content",
    "Encouraging community interaction and engagement",
    "Fake reviews or testimonials",
    "Political bias or agenda", // 4
    "Harmful stereotypes or generalizations",
    "Fair use, copyright, and intellectual property rights",
    "Invasive or disturbing content",
    "Emotional manipulation or fearmongering",
    "Discrimination or exclusion",
    "Ethical advertising practices", // 1
    "Respecting personal boundaries and consent",
    "Solutions-focused reporting",
    "Audience participation and feedback",
    "Promotion of scientific literacy and skepticism",
    "Misuse of personal information",
    "Responsible consumption and production",
    "False equivalency or balance fallacy",
    "Lack of context or missing information",
    "Promotion of physical health and fitness",
    "No direct or indirect discrimination",
    "Demonstrably false information",
    "Exploitative or predatory practices",
    "Insensitive or offensive language",
    "Encouraging responsible behavior",
    "To exploit vulnerable populations",
    "Violent or harmful ideologies",
    "Sexual exploitation or objectification",
    "Unwarranted personal data collection",
    "Unverified rumors or conspiracy theories",
    "Harassment or cyberbullying",
    "Trivializing serious issues",
    "Trustworthy, reputable endorsements",
    "Avoidance of scapegoating or blame-shifting",
    "Promoting community and social cohesion",
    "Incitement of panic or disorder",
    "Sustainable business and economic practices",
    "Undisclosed paid endorsements or reviews",
    "Political neutrality or non-partisanship", // 10
    "Professional ethics and codes of conduct",
    "Sentiment manipulation or astroturfing",
    "Suspicious, potential scams or fraud",
    "Get-rich-quick schemes or unrealistic promises", // 20
    "Aggressive, high-pressure sales techniques",
    "Deliberate obfuscation or confusion",
    "Promotion of balanced, diverse viewpoints",
    "Sources with a history of inaccurate information", // 19
    "Encouraging civic engagement and participation",
    "Demonization or dehumanization of individuals or groups",
    "Promotion of tolerance and understanding",
    "Invasive tracking or surveillance",
    "Crisis or disaster exploitation", // 23
    "Healthy online etiquette and behavior",
    "Explicit content without appropriate warnings",
    "Distorting facts or cherry-picking data",
    "Promotion of active, healthy lifestyles",
    "Use of emotive or charged language to manipulate",
    "Public accountability and transparency",
    "Exploitation of personal insecurities",
    "Perpetuation of harmful or outdated stereotypes",
    "Encouraging lifelong learning and curiosity",
    "Undisclosed affiliate links or partnerships", // 14
    "Disrespect of cultural traditions or practices",
    "Promotion of community service and volunteerism",
    "Fake social proof or manipulated metrics",
    "Fueling divisive or polarizing debates", // 8
    "Deceptive appearance of objectivity",
    "Protection of intellectual property and creative rights",
    "Invasion of personal privacy or doxxing",
    "Instigation of hatred or prejudice", // 11
    "Discouraging active participation in society",
    "Non-transparent data use practices",
    "Denial or disregard of established facts or consensus",
    "Encouraging creative and critical thinking",
    "Accountability for false or misleading claims",
    "Promotion of misinformation under guise of 'alternative facts'", // 15
    "Insensitive or offensive humor",
    "Promoting a culture of respect and dignity",
    "Fueling or perpetuating conspiracy theories",
    "Lack of diversity in perspectives or representation", // 12
    "Promotion of responsible digital citizenship", // 2
    "Normalization of harmful behaviors or attitudes",
    "Unwarranted undermining of reputable institutions or experts",
    "Promotion of harmful pseudoscientific beliefs or practices",
    "Uncritical amplification of controversial or extremist views",
    "Uncritical sharing of unverified information",
    "Clickbait, suspected spam, fake news, sensationalism, hype",
    "Authentic, verified news/information",
    "Aggressive marketing, advertising, selling, promotion, authoritative, commanding", // 7
    "Informative content, unbiased information",
    "Call to immediate action", // 13
    "No urgency or pressure to take action, passive suggestion",
    "Suspicious, questionable, dubious", // 9
    "Trustworthy, credible, reliable",
    "Untrustworthy, not to be trusted, unreliable source, blacklisted", // 3
    "Reputable source", // 5
    "Misleading or deceptive information: The product advertisement made false claims about the benefits of the product.", // 6
    "Accurate, transparent information",
    "Of importance, significant, crucial",
    "Insignificant, inconsequential", // 17
    "Giveaway, tokens, airdrops, rewards, gratis, claim now", // 16
    "No incentives or rewards provided",
    "To hide illegal activity",
    "Legal, lawful activity",
    "Exaggeration or hyperbole",
    "Factual, restrained language",
    "Sensationalism in headlines",
    "Balanced, informative headlines",
    "Bias or slant", // 24
    "Objective, unbiased reporting",
    "Editorial or opinion pieces",
    "Fact-based reporting",
    "Unverified or unverified content",
    "Fact-checking or verification",
    "Sponsored content or native advertising",
    "Independent, non-sponsored content",
    "User-generated content",
    "Professional journalism or organization-created content", // 22
    "Comparing reputation, bias, credibility",
    "News sources or media outlets",
    "Irresponsible consumption and ecological degradation",
    "Sustainable practices and environmental impact",
    "Harassment/threatening",
    "Constructive communication",
    "Violence",
    "Peaceful behavior",
    "Sexual",
    "Non-sexual in nature",
    "Hate",
    "Expressing kindness and acceptance",
    "Self-harm",
    "Promoting well-being and self-care",
    "Sexual/minors",
    "Content appropriate for all ages",
    "Hate/threatening", // 18
    "Positive and supportive communication",
    "Violence/graphic",
    "Non-violent and non-graphic",
    "Self-harm/intent",
    "Encouraging positive intentions",
    "Self-harm/instructions",
    "Promoting safety and well-being"
];

pub const FRAUD_INDICATORS: [&str;24] = [
    //"Reliable, trustworthy references or sources",
    //"Neutral language, no leading statements",
    //"Educational content",
    //"Promotion of inclusivity and diversity",
    //"Censorship or freedom of speech issues",
    //"Healthy debate and exchange of ideas",
    //"Appropriate content rating for audience",
    //"No personal attacks or defamatory language",
    //"Promotion of mental health awareness",
    //"Ethical sourcing or supply chain",
    //"Public interest or societal impact",
    //"Use of data and privacy considerations",
    //"Intentional disinformation or propaganda",
    //"Promotion of literacy and critical thinking",
    "Disclosure of conflicts of interest", // 21
    //"Culturally sensitive content",
    //"Encouraging community interaction and engagement",
    //"Fake reviews or testimonials",
    "Political bias or agenda", // 4
    //"Harmful stereotypes or generalizations",
    //"Fair use, copyright, and intellectual property rights",
    //"Invasive or disturbing content",
    //"Emotional manipulation or fearmongering",
    //"Discrimination or exclusion",
    "Ethical advertising practices", // 1
    //"Respecting personal boundaries and consent",
    //"Solutions-focused reporting",
    //"Audience participation and feedback",
    //"Promotion of scientific literacy and skepticism",
    //"Misuse of personal information",
    //"Responsible consumption and production",
    //"False equivalency or balance fallacy",
    //"Lack of context or missing information",
    //"Promotion of physical health and fitness",
    //"No direct or indirect discrimination",
    //"Demonstrably false information",
    //"Exploitative or predatory practices",
    //"Insensitive or offensive language",
    //"Encouraging responsible behavior",
    //"To exploit vulnerable populations",
    //"Violent or harmful ideologies",
    //"Sexual exploitation or objectification",
    //"Unwarranted personal data collection",
    //"Unverified rumors or conspiracy theories",
    //"Harassment or cyberbullying",
    //"Trivializing serious issues",
    //"Trustworthy, reputable endorsements",
    //"Avoidance of scapegoating or blame-shifting",
    //"Promoting community and social cohesion",
    //"Incitement of panic or disorder",
    //"Sustainable business and economic practices",
    //"Undisclosed paid endorsements or reviews",
    "Political neutrality or non-partisanship", // 10
    //"Professional ethics and codes of conduct",
    //"Sentiment manipulation or astroturfing",
    //"Suspicious, potential scams or fraud",
    "Get-rich-quick schemes or unrealistic promises", // 20
    //"Aggressive, high-pressure sales techniques",
    //"Deliberate obfuscation or confusion",
    //"Promotion of balanced, diverse viewpoints",
    "Sources with a history of inaccurate information", // 19
    //"Encouraging civic engagement and participation",
    //"Demonization or dehumanization of individuals or groups",
    //"Promotion of tolerance and understanding",
    //"Invasive tracking or surveillance",
    "Crisis or disaster exploitation", // 23
    //"Healthy online etiquette and behavior",
    //"Explicit content without appropriate warnings",
    //"Distorting facts or cherry-picking data",
    //"Promotion of active, healthy lifestyles",
    //"Use of emotive or charged language to manipulate",
    //"Public accountability and transparency",
    //"Exploitation of personal insecurities",
    //"Perpetuation of harmful or outdated stereotypes",
    //"Encouraging lifelong learning and curiosity",
    "Undisclosed affiliate links or partnerships", // 14
    //"Disrespect of cultural traditions or practices",
    //"Promotion of community service and volunteerism",
    //"Fake social proof or manipulated metrics",
    "Fueling divisive or polarizing debates", // 8
    //"Deceptive appearance of objectivity",
    //"Protection of intellectual property and creative rights",
    //"Invasion of personal privacy or doxxing",
    "Instigation of hatred or prejudice", // 11
    //"Discouraging active participation in society",
    //"Non-transparent data use practices",
    //"Denial or disregard of established facts or consensus",
    //"Encouraging creative and critical thinking",
    //"Accountability for false or misleading claims",
    "Promotion of misinformation under guise of 'alternative facts'", // 15
    //"Insensitive or offensive humor",
    //"Promoting a culture of respect and dignity",
    //"Fueling or perpetuating conspiracy theories",
    "Lack of diversity in perspectives or representation", // 12
    "Promotion of responsible digital citizenship", // 2
    //"Normalization of harmful behaviors or attitudes",
    //"Unwarranted undermining of reputable institutions or experts",
    //"Promotion of harmful pseudoscientific beliefs or practices",
    //"Uncritical amplification of controversial or extremist views",
    //"Uncritical sharing of unverified information",
    //"Clickbait, suspected spam, fake news, sensationalism, hype",
    //"Authentic, verified news/information",
    "Aggressive marketing, advertising, selling, promotion, authoritative, commanding", // 7
    //"Informative content, unbiased information",
    "Call to immediate action", // 13
    //"No urgency or pressure to take action, passive suggestion",
    "Suspicious, questionable, dubious", // 9
    //"Trustworthy, credible, reliable",
    "Untrustworthy, not to be trusted, unreliable source, blacklisted", // 3
    "Reputable source", // 5
    "Misleading or deceptive information: The product advertisement made false claims about the benefits of the product.", // 6
    //"Accurate, transparent information",
    //"Of importance, significant, crucial",
    "Insignificant, inconsequential", // 17
    "Giveaway, tokens, airdrops, rewards, gratis, claim now", // 16
    //"No incentives or rewards provided",
    //"To hide illegal activity",
    //"Legal, lawful activity",
    //"Exaggeration or hyperbole",
    //"Factual, restrained language",
    //"Sensationalism in headlines",
    //"Balanced, informative headlines",
    "Bias or slant", // 24
    //"Objective, unbiased reporting",
    //"Editorial or opinion pieces",
    //"Fact-based reporting",
    //"Unverified or unverified content",
    //"Fact-checking or verification",
    //"Sponsored content or native advertising",
    //"Independent, non-sponsored content",
    //"User-generated content",
    "Professional journalism or organization-created content", // 22
    //"Comparing reputation, bias, credibility",
    //"News sources or media outlets",
    //"Irresponsible consumption and ecological degradation",
    //"Sustainable practices and environmental impact",
    //"Harassment/threatening",
    //"Constructive communication",
    //"Violence",
    //"Peaceful behavior",
    //"Sexual",
    //"Non-sexual in nature",
    //"Hate",
    //"Expressing kindness and acceptance",
    //"Self-harm",
    //"Promoting well-being and self-care",
    //"Sexual/minors",
    //"Content appropriate for all ages",
    "Hate/threatening", // 18
    //"Positive and supportive communication",
    //"Violence/graphic",
    //"Non-violent and non-graphic",
    //"Self-harm/intent",
    //"Encouraging positive intentions",
    //"Self-harm/instructions",
    //"Promoting safety and well-being"
];

pub fn create_training_data(dataset_paths: Vec<&str>,topics_output_path: &str) -> anyhow::Result<()> {

    let mut dataset: Vec<(String,f64)> = read_datasets(&dataset_paths)?;


    let spam_count = dataset.iter().filter(|x| x.1 >= 0.5).count();
    let ham_count = dataset.iter().filter(|x| x.1 < 0.5).count();
    let total_count = dataset.len();

    let spam_percentage = (spam_count as f64 / total_count as f64) * 100.0;
    let ham_percentage = (ham_count as f64 / total_count as f64) * 100.0;

    println!("Spam count: {}", spam_count);
    println!("Ham count: {}", ham_count);
    println!("===================");
    println!("Total count: {}", total_count);
    println!("===================");
    println!("Spam percentage: {:.2}%", spam_percentage);
    println!("Ham percentage: {:.2}%\n", ham_percentage);

    /*
    let indices_spam: Vec<usize> = dataset.iter().enumerate().filter(|(i,x)| x.1).map(|(i,_)| i).collect();
    let indices_spam_ham: Vec<usize> = dataset.iter().enumerate().filter(|(i,x)| !x.1)/*.take(indices_spam.len())*/.map(|(i,_)| i).collect();
    let dataset: Vec<(&str,bool)> = vec![
        indices_spam.iter().map(|&i| (dataset[i].0.as_str(),dataset[i].1)).collect::<Vec<(&str,bool)>>(),
        indices_spam_ham.iter().map(|&i| (dataset[i].0.as_str(),dataset[i].1)).collect::<Vec<(&str,bool)>>()
    ].into_iter().flatten().map(|x| x.clone()).collect();
    */


    let dataset_view: Vec<(&str,&f64)> = dataset.iter().map(|(text,label)| (text.as_str(),label)).collect();


    sentiment::extract_sentiments(&dataset_view,Some(format!("sentiment_extract_sentiments_{}",topics_output_path)))?;
    language_model::extract_topics(&dataset_view,&FRAUD_INDICATORS,Some(format!("language_model_extract_topics_{}",topics_output_path)))?;

    Ok(())
}

pub fn create_classification_model(x_dataset_shuffled: &Vec<Vec<f64>>, y_dataset_shuffled: &Vec<f64>) -> anyhow::Result<()> {

    classification::update_regression_model(&x_dataset_shuffled, &y_dataset_shuffled)?;

    Ok(())
}
pub fn test_classification_model(x_dataset_shuffled: &Vec<Vec<f64>>, y_dataset_shuffled: &Vec<f64>) -> anyhow::Result<()> {

    classification::test_regression_model(&x_dataset_shuffled, &y_dataset_shuffled)?;

    Ok(())
}

pub fn create_naive_bayes_model(train_dataset: &Vec<(String,f64)>, test_dataset: &Vec<(String,f64)>) -> anyhow::Result<()> {

    let x_dataset = train_dataset.iter().map(|x| x.0.to_string()).collect::<Vec<String>>();
    let y_dataset = train_dataset.iter().map(|x| if x.1 < 0.5 { 0 } else { 1 }).collect::<Vec<i32>>();

    let test_x_dataset = test_dataset.iter().map(|x| x.0.to_string()).collect::<Vec<String>>();
    let test_y_dataset = test_dataset.iter().map(|x| if x.1 < 0.5 { 0 } else { 1 }).collect::<Vec<i32>>();

    naive_bayes::update_naive_bayes_model(&x_dataset,&y_dataset,&test_x_dataset,&test_y_dataset)?;
    naive_bayes::update_categorical_naive_bayes_model(&x_dataset,&y_dataset,&test_x_dataset,&test_y_dataset)?;
    //classification::test_linear_regression_model(&x_dataset,&y_dataset)?;

    Ok(())
}
