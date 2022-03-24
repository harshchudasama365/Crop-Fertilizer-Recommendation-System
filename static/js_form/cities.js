var state_arr = new Array("Andaman & Nicobar", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Dadra & Nagar Haveli", "Daman & Diu", "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Orissa", "Pondicherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Tripura", "Uttar Pradesh", "Uttaranchal", "West Bengal");

var s_a = new Array();

s_a[0]="";
s_a[1]=" Andaman Island | Beadonabad | Malappuram | Nicobar Island ";
s_a[2]=" Adilabad | Anantapur | Asifabad | Chittoor  | Cuddapah | East Godavari | Gadwal | Godavari | Guntur | Hyderabad  | Jagtial | Jangaon | Karimnagar | Khammam | Kothagudem  | Krishna | Kurnool | Nagarkurnool | Narayanpet | Nellore | Nirmal | Nizamabad | Kondagaon  | Peddapalli | Rangareddy | Sangareddy | Siddipet  | Sircilla | Srikakulam | kota | Visakhapatnam  | Vizianagaram  | Wanaparthy | Warangal ";
s_a[3]=" Anjaw | Changlang | Dibang Valley | Kameng | Kurung Kumey | Lohit | Lower Dibang Valley | Lower Subansiri | Papum Pare | Siang | Tawang | Tirap | Upper Siang | Upper Subansiri ";
s_a[4]=" Barpeta Road | Bongaigaon | Cachar | Cachar Hills | Darrang  | Dhemaji | Dhubri | Dibrugarh  | Goalpara  | Golaghat  | Hailakandi | Jorhat | Kamrup | Karbi Anglong | Karimganj | Kokrajhar | Lakhimpur | Morigaon | Nagaon | Nalbari | Sibsagar | Sonitpur | Tinsukia  | Udalguri ";
s_a[5]=" Araria | Arwal | Aurangabad  | Banka | Begusarai | Bhagalpur | Bhojpur | Buxar  | Champaran | Darbhanga | Gaya | Gopalganj | Jahanabad | Jamui | Kaimur (Bhabua) | Katihar | Kishanganj | Lakhisarai | Madhepura | Madhubani | Maharajganj | Munger | Muzaffarpur | Patna | Purnia | Ramnagar | Rohtas   | Saharsa | Samastipur | Saran | Sheikhpura | Sitamarhi | Siwan | Sonepur | Supaul | Vaishali ";
s_a[6]=" Chandigarh ";
s_a[7]=" Balod | Balodabazar | Balrampur | Bijapur | Bilaspur | Dantewada | Dhamtari | Durg | Gariaband | Jashpur  | Kabirdham-Kawardha | Kanker | Kawardha | Kondagaon  | Korba | Korea | Kota | Mahasamund | Mungeli | Narayanpur | Pali | Patan | Raigarh | Raipur | Rajnandgaon | Sitapur | Sukma | Surajpur | Surguja | Udaipur  ";
s_a[8]=" Dadra & Nagar Haveli ";
s_a[9]=" Daman | Diu ";
s_a[10]=" Delhi | East Delhi | New Delhi | North Delhi | North East Delhi | North West Delhi | South Delhi | South West Delhi ";
s_a[11]=" Candolim | Goa ";
s_a[12]=" Ahmedabad | Amreli | Anand | Banaskantha | Bharuch | Bhavnagar | Botad  | Chhota Udaipur | Dahod  | Dang | Dwarka | Gandhi Nagar | Jamnagar | Junagarh | Kutch | Mehsana | Morvi  | Narmada  | Navasari | Panchmahals | Patan  | Porbandar | Rajkot   | Sabarkantha  | Surat | Surendranagar  | Vadodara    | Valsad ";
s_a[13]=" Bhiwani | Faridabad | Fatehabad | Gurgaon  | Hisar   | Jhajjar | Jind  | Kaithal    | Karnal  | Kurukshetra  | Mahendragarh  | Mewat | Palwal | Panchkula | Panipat | Rewari | Rohtak  | Sirsa  | Sonipat | Yamunanagar ";
s_a[14]=" Bilaspur | Chamba | Hamirpur | Kangra | Kinnaur | Kullu  | Mandi | Rajgarh | Shimla | Sirmaur | Solan | Udaipur | Una";
s_a[15]=" Anantnag | Badgam | Bandipur | Baramulla  darwah  | Doda  | Jammu  | Kargil  | Kathua | Kishtwar | Kulgam | Kupwara | Leh | Poonch | Pulwama | Rajouri | Ramban | Ramnagar | Reasi | Samba | Srinagar | Udhampur ";
s_a[16]=" Bokaro | Chatra | Deoghar | Dhanbad | Dumka  | Garhwa  | Giridih | Godda   | Gumla | Hazaribagh | Jamtara | Kharsawa | Khunti | Koderma | Latehar | Lohardaga | Pakur | Palamu  | Patan | Ramgarh | Ranchi | Sahibganj | Simdega | Singhbhum  ";
s_a[17]=" Bagalkot | Bangalore | Bangalore Rural | Belgaum |  Bellary | Belthangady | Bidar | Bijapur | Chikkaballapur | Chikmagalur | Chitradurga | Davanagere | Dharwad  | Gadag | Gulbarga | Hassan | Haveri | Kodagu | Kolar | Mandya | Mysore | Raichur | Sagar | Shimoga | Tumkur  | Udupi | Yadgiri  ";
s_a[18]=" Alappuzha  | Ernakulam | Idukki | Kannur | Kollam | Kottayam  | Kozhikode  | Malappuram | Palakkad   | Pathanamthitta | Thiruvananthapuram  | Thrissur | Wayanad ";
s_a[19]=" Andaman Island | Lakshadweep Sea ";
s_a[20]=" Agar | Alirajpur | Anuppur | Ashoknagar | Balaghat | Banda | Barwani | Bhind  | Bhopal | Burhanpur | Chhindwara | Damoh | Datia | Dewas | Dhar  | Dindori | Gopalganj | Guna  | Gwalior   | Harda  | Hoshangabad  | Indore | Jabalpur | Jhabua | Katni | Khandwa  | Khargone | Mandla | Mandsaur | Morena | Narsinghpur | Neemuch | Panna  | Patan | Raisen | Rajgarh  | Ratlam  | Rewa  | Sagar | Satna  | Sehore   | Seoni  | Shahdol   | Shajapur pur  | Shivpuri  | Sidhi | Singrauli | Tikamgarh | Ujjain | Umaria  | Vidisha ";
s_a[21]=" Ahmednagar | Akola | Amravati | Aurangabad | Bhandara  | Buldhana  | Chandrapur | Delhi Tanda | Dhule | Dindori | Gondia | Hingoli | Jalgaon | Jalna | Kolhapur | Latur | Mumbai | Nagpur  | Nanded | Nandurbar  | Nashik | Osmanabad | Palghar | Pali | Parbhani | Patan | Pune | Raigad | Ratnagiri | Sangli | Satara | Sholapur  | Sindhudurg | Thane  | Wardha | Washim  | Yavatmal ";
s_a[22]=" Bishnupur  | Chandel  | Churachandpur | Imphal | Senapati | Tamenglong  | Thoubal | Ukhrul ";
s_a[23]=" Garo Hills | Jaintia Hills | Khasi Hills ";
s_a[24]=" Aizawl | Champhai | Kolasib | Lawngtlai | Lunglei | Mamit | Saiha | Serchhip";
s_a[25]=" Dimapur  | Kiphire | Kohima | Mokokchung | Mon | Phek | Tuensang | Wokha | Zunheboto ";
s_a[26]=" Angul | Balangir | Balasore | Bargarh | Boudh | Cuttack | Deogarh | Dhenkanal | Gajapati | Ganjam | Jajpur | Jharsuguda  | Kalahandi | Kandhamal  | Kendrapara  | Keonjhar   | Khurda  | Koraput | Malkangiri  | Mayurbhanj   | Nabarangapur | Narsinghpur | Nayagarh   | Nowrangapur | Nuapada | Puri | Rayagada  | Sambalpur ";
s_a[27]=" Karaikal | Mahe | Pondicherry | Yanam ";
s_a[28]=" Amritsar | Barnala  | Bathinda | Chandigarh | Faridkot | Fatehgarh Sahib | Gurdaspur | Hoshiarpur | Jalandhar | Kapurthala | Ludhiana | Mansa | Moga | Muktasar | Patiala | Ropar | Sangrur | SAS Nagar | Urmar | Tarn Taran ";
s_a[29]=" Ajmer | Alwar | Banswara | Baran | Barmer | Bharatpur | Bhilwara   | Bikaner  | Bundi | Chittorgarh  | Churu ramgarh | Dausa | Deogarh | Dholpur | Dungarpur  | Fatehpur | Hanumangarh | Jaipur | Jaisalmer | Jalore | Jhalawar | Jhunjhunu | Jodhpur | Karauli | Kishanganj | Kota  |  Nagaur | Pali | Pratapgarh | Raipur  | Rajgarh | Rajsamand  | Ramgarh | Sawai Madhopur | Sikar | Sirohi | Sri Ganganagar | Tonk | Udaipur ";
s_a[31]=" Ariyalur | Chengalpattu  | Chennai  | Coimbatore  | Cuddalore | Dharmapuri | Dindigul | Erode | Kanchipuram  | Kanyakumari | Karaikal  | Karur | Krishnagiri | Madurai | Nagapattinam  | Namakkal    | Nilgiris  | Perambalur    | Pondicherry | Pudukkottai | Ranipet  | Salem | Sivaganga    | Tenkasi | Thanjavur | Theni   | Thoothukudi | Tirunelvelli | Tirupathur | Tirupur  | Tiruvallur | Tiruvannamalai | Tiruvarur   | Trichy | Tuticorin | Vellore   | Villupuram  | Virudhunagar  ";
s_a[32]=" Dhalai | Tripura ";
s_a[33]=" Agra   | Aligarh | Allahabad | Auraiya | Azamgarh  | Badaun | Baghpat | Bahraich | Ballia | Balrampur | Banda | Barabanki | Bareilly | Basti | Bijnore | Bulandshahr | Chandauli | Chitrakoot | Etah | Etawah | Faizabad | Farrukhabad | Fatehpur | Firozabad | Gautam Buddha Nagar  | Ghaziabad | Ghazipur  | Gonda | Gorakhpur | Hamirpur | Hapur | Hardoi | Hathras  | Jalaun  | Jaunpur | Jhansi | Jyotiba Phule Nagar | Kannauj | Kanpur | Kaushambi | Kheri | Kushinagar  | Lalitpur | Lucknow  | Maharajganj | Mahoba | Mainpuri | Mau | Meerut | Mirzapur | Moradabad  | Muzaffarnagar | Pilibhit   | Pratapgarh   | Raibareli | Rampur| Saharanpur | Sambhal  | Sant Kabir Nagar | Sant Ravidas Nagar| Shahjahanpur  | Shravasti | Siddharthnagar | Sitapur  | Sonbhadra  | Sultanpur    | Unnao  | Varanasi ";
s_a[34]=" Almora | Bageshwar | Chamoli | Champawat | Dehradun | Haridwar | Nainital  | Pauri Garhwal | Pithoragarh  | Rajgarh  | Rudraprayag | Tehri Garhwal | Udham Singh Nagar  | Uttarkashi ";
s_a[35]=" Bankura  | Birbhum  | Burdwan | Calcutta | Cooch Behar| Darjeeling   | Dinajpur | Hooghly | Howrah   | Jalpaiguri  | Malda  abhanga | Medinipur  | Mirzapur | Murshidabad | Nadia | Nayagarh | Parganas | Purulia ";

function print_state(state_id){
	// given the id of the <select> tag as function argument, it inserts <option> tags
	var option_str = document.getElementById(state_id);
	option_str.length=0;
	option_str.options[0] = new Option('Select State','');
	option_str.selectedIndex = 0;
	for (var i=0; i<state_arr.length; i++) {
		option_str.options[option_str.length] = new Option(state_arr[i],state_arr[i]);
	}
}

function print_city(city_id, city_index){
	var option_str = document.getElementById(city_id);
	option_str.length=0;	// Fixed by Julian Woods
	option_str.options[0] = new Option('Select City','');
	option_str.selectedIndex = 0;
	var city_arr = s_a[city_index].split("|");
	for (var i=0; i<city_arr.length; i++) {
		option_str.options[option_str.length] = new Option(city_arr[i],city_arr[i]);
	}
}


// var state_arr = new Array("Andaman & Nicobar", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Dadra & Nagar Haveli", "Daman & Diu", "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Orissa", "Pondicherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Tripura", "Uttar Pradesh", "Uttaranchal", "West Bengal");
// var s_a = new Array();
// s_a[0]=""
// s_a[1]=" Andaman Island | Malappuram | Yadita";
// s_a[2]=" Adilabad | Anantapur | Chittoor | Cuddapah | East Godavari | Godavari | Guntur | Hyderabad | Jagtial | Jangaon | Karimnagar | Khammam | Krishna | Kurnool | Mahabubabad | Mancherial | Medak | Nagarkurnool | Nalgonda | Narayanpet | Nellore | Nirmal | Nizamabad | Prakasam | Rangareddy | Sangareddy | Siddipet | Sircilla | Srikakulam | Visakhapatnam | Vizianagaram | Wanaparthy | Warangal | Zahirabad ";
// s_a[3]=" Anjaw | Changlang | Dibang Valley | Kameng | Kurung Kumey | Lohit | Lower Dibang Valley | Lower Subansiri | Siang | Tawang | Tirap | Upper Siang | Upper Subansiri | Yiang Kiag ";
// s_a[4]=" Bongaigaon | Cachar | Cachar Hills | Darrang | Dhemaji | Dhubri | Dibrugarh | Goalpara | Golaghat | Hailakandi | Jorhat | Karbi Anglong | Karimganj | Kokrajhar | Lakhimpur | Morigaon | Nagaon | Nalbari | Sibsagar | Sonitpur | Tinsukia | Udalguri | UdarbondhBarpeta";
// s_a[5]=" Araria | Arwal | Aurangabad | Banka | Begusarai | Bhagalpur | Bhojpur | Buxar | Champaran | Darbhanga | Gaya | Gopalganj | H.Jahanabad | Jamui | Kaimur (Bhabua) | Katihar | Khagaria | Kishanganj | Lakhisarai | Madhepura | Madhubani | Maharajganj | Munger | Muzaffarpur | Nalanda | Nawada | Patna | Purnia | Ramnagar | Rohtas | Saharsa | Samastipur | Saran | Sheikhpura | Sheohar | Sitamarhi | Siwan | Sonepur | Supaul | Vaishali | Wazirganj";\ns_a[6]=" Chandigarh | Mani Marja";\ns_a[7]=" Balod | Balodabazar | Balrampur | Bastar | Bemetara | Bijapur | Bilaspur | Dantewada | Dhamtari | Durg | Gariaband | Janjgir-Champa | Jashpur | Kanker | Kawardha | Kondagaon | Korba | Kota | Mahasamund | Mungeli | Narayanpur | Pali | Patan | Raigarh | Raipur | Rajnandgaon | Sitapur | Sukma | Surajpur | Surguja | Udaipur | Wadrainagar";\ns_a[8]=" Dadra & Nagar Haveli | Velugam ";\ns_a[9]=" Daman | Diu | Passo Covo ";\ns_a[10]=" East Delhi | New Delhi | North Delhi | North East Delhi | North West Delhi | South Delhi | South West Delhi | West Delhi ";\ns_a[11]=" Goa | Terekhol ";\ns_a[12]=" Ahmedabad | Amreli | Anand | Banaskantha | Bhavnagar | Botad | Dahod | Dwarka | Jamnagar | Junagarh | Kheda | Kutch | Mehsana | Narmada | Navasari | Panchmahals | Patan | Porbandar | Rajkot | Sabarkantha | Surat | Surendranagar | Vadodara | Valsad | Wankaner ";\ns_a[13]=" Ambala | Bhiwani | Faridabad | Fatehabad | Gurgaon | Hisar | Jhajjar | Jind | Kaithal | Karnal | Kurukshetra | Mahendragarh | Mewat | Palwal | Panchkula | Panipat | Rewari | Rohtak | Sirsa | Sonipat | Yamunanagar ";\ns_a[14]=" Bilaspur | Chamba | Hamirpur | Kangra | Kinnaur | Kullu | Mandi | Rajgarh | Shimla | Sirmaur | Solan | Udaipur | Una";\ns_a[15]=" Anantnag | Badgam | Bandipur | Baramulla | Doda | Jammu | Kargil | Kathua | Kishtwar | Kulgam | Kupwara | Leh | Poonch | Pulwama | Rajouri | Ramban | Ramnagar | Reasi | Samba | Srinagar | Udhampur | Vaishno Devi ";\ns_a[16]=" Bokaro | Chatra | Deoghar | Dhanbad | Dumka | Garhwa | Giridih | Godda | Gumla | Hazaribagh | Jamtara | Khunti | Koderma | Latehar | Lohardaga | Pakur | Palamu | Patan | Ramgarh | Ranchi | Sahibganj | Simdega | Singhbhum | Torpa ";\ns_a[17]=" Bangalore Rural | Belgaum | Bellary | Bidar | Bijapur | Chikmagalur | Chitradurga | Davanagere | Dharwad | Gadag | Gulbarga | Hassan | Haveri | Kannada | Kodagu | Kolar | Koppal | Mandya | Mysore | Raichur | Sagar | Shimoga | Tumkur | Udupi | Yadgiri | Yellapur ";\ns_a[18]=" Alappuzha | Ernakulam | Idukki | Kannur | Kollam | Kottayam | Kozhikode | Malappuram | Palakkad | Pathanamthitta | Thiruvananthapuram | Thrissur | Wayanad ";\ns_a[19]=" Lakshadweep Sea | South Island ";\ns_a[20]=" Agar | Alirajpur | Anuppur | Ashoknagar | Balaghat | Banda | Barwani | Betul | Bhind | Bhopal | Burhanpur | Chhatarpur | Chhindwara | Damoh | Datia | Dewas | Dhar | Dindori | Gopalganj | Guna | Gwalior | Harda | Hoshangabad | Indore | Jabalpur | Jhabua | Katni | Khandwa | Khargone | Mandla | Mandsaur | Morena | Narsinghpur | Neemuch | Panna | Patan | Raisen | Rajgarh | Ratlam | Rewa | Sagar | Satna | Sehore | Seoni | Shahdol | Shajapur | Sheopur | Shivpuri | Sidhi | Singrauli | Tikamgarh | Ujjain | Umaria | Vidisha | Zhirnia ";\ns_a[21]=" Ahmednagar | Akola | Amravati | Aurangabad | Beed | Bhandara | Buldhana | Chandrapur | Delhi Dhule | Dindori | Gondia | Hingoli | Jalgaon | Jalna | Kolhapur | Latur | Mumbai | Nagpur | Nanded | Nandurbar | Osmanabad | Palghar | Pali | Parbhani | Patan | Pune | Raigad | Ratnagiri | Sangli | Satara | Sindhudurg | Thane | Wardha | Washim | Yavatmal | Yeotmal ";\ns_a[22]=" Bishnupur | Chandel | Churachandpur | Imphal | Senapati | Tamenglong | Thoubal | Ukhrul ";\ns_a[23]=" Garo Hills | Jaintia Hills | Khasi Hills | Ri Bhoi | Williamnagar";\ns_a[24]=" Champhai | Kolasib | Lawngtlai | Lunglei | Mamit | Saiha | Serchhip";\ns_a[25]=" Dimapur | Kiphire | Kohima | Mokokchung | Mon | Phek | Tuensang | Wokha | Zunheboto ";\ns_a[26]=" Angul | Balasore | Bhadrak | Boudh | Cuttack | Deogarh | Dhenkanal | Gajapati | Ganjam | Jajpur | Jharsuguda | Kalahandi | Kandhamal | Kendrapara | Keonjhar | Khurda | Koraput | Malkangiri | Mayurbhanj | Narsinghpur | Nayagarh | Nuapada | Puri | Rayagada | Sambalpur | Umerkote ";\ns_a[27]=" Karaikal | Mahe | Pondicherry | Yanam ";\ns_a[28]=" Amritsar | Barnala | Bathinda | Chandigarh | Faridkot | Fatehgarh Sahib | Gurdaspur | Hoshiarpur | Jalandhar | Kapurthala | Ludhiana | Mansa | Moga | Nawanshahar | Patiala | Ropar | Sangrur | Tarn Taran | Zira ";\ns_a[29]=" Ajmer | Alwar | Banswara | Baran | Barmer | Bharatpur | Bhilwara | Bikaner | Bundi | Chittorgarh | Churu | Dausa | Deogarh | Dholpur | Dungarpur | Fatehpur | Hanumangarh | Jaipur | Jaisalmer | Jalore | Jhalawar | Jhunjhunu | Jodhpur | Karauli | Kishanganj | Kota | Nagaur | Pali | Pratapgarh | Raipur | Rajgarh | Rajsamand | Ramgarh | Sawai Madhopur | Sikar | Sirohi | Sri Ganganagar | Tonk | Udaipur | Viratnagar ";\ns_a[30]=" Yumtang ";\ns_a[31]=" Ariyalur | Chengalpattu | Chennai | Coimbatore | Cuddalore | Dharmapuri | Dindigul | Erode | Kanchipuram | Kanyakumari | Karaikal | Karur | Krishnagiri | Madurai | Nagapattinam | Namakkal | Nilgiris | Perambalur | Pondicherry | Pudukkottai | Ramanathapuram | Ranipet | Salem | Sivaganga | Tenkasi | Thanjavur | Theni | Tirunelvelli | Tirupathur | Tiruvallur | Tiruvannamalai | Tiruvarur | Trichy | Tuticorin | Vellore | Villupuram | Virudhunagar | Yercaud ";\ns_a[32]=" Dhalai | Tripura ";\ns_a[33]=" Agra | Aligarh | Allahabad | Ambedkar Nagar | Amethi | Auraiya | Azamgarh | Badaun | Baghpat | Bahraich | Ballia | Balrampur | Banda | Barabanki | Bareilly | Basti | Bulandshahr | Chandauli | Chitrakoot | Deoria | Etah | Etawah | Faizabad | Farrukhabad | Fatehpur | Firozabad | Ghaziabad | Ghazipur | Gonda | Gorakhpur | Hamirpur | Hapur | Hardoi | Hathras | Jalaun | Jaunpur | Jhansi | Jyotiba Phule Nagar | Kannauj | Kanpur | Kaushambi | Kheri | Lalitpur | Lucknow | Maharajganj | Mahoba | Mainpuri | Mathura | Mau | Meerut | Mirzapur | Moradabad | Muzaffarnagar | Pilibhit | Pratapgarh | Rampur | Saharanpur | Sambhal | Sant Kabir Nagar | Sant Ravidas Nagar | Shahjahanpur | Shravasti | Sitapur | Sonbhadra | Sultanpur | Unnao | Varanasi | Zamania ";\ns_a[34]=" Almora | Bageshwar | Chamoli | Champawat | Dehradun | Haridwar | Nainital | Pauri Garhwal | Pithoragarh | Rajgarh | Rudraprayag | Udham Singh Nagar | Uttarkashi ";\ns_a[35]=" Bankura | Birbhum | Burdwan | Calcutta | Cooch Behar | Darjeeling | Dinajpur | Hooghly | Howrah | Jalpaiguri | Malda | Mirzapur | Murshidabad | Nadia | Nayagarh | Parganas | Purulia | Tamluk";\n\nfunction print_state(state_id){\n\t// given the id of the <select> tag as function argument, it inserts <option> tags\n\tvar option_str = document.getElementById(state_id);\n\toption_str.length=0;\n\toption_str.options[0] = new Option(\'Select State\',\'\');\n\toption_str.selectedIndex = 0;\n\tfor (var i=0; i<state_arr.length; i++) {\n\t\toption_str.options[option_str.length] = new Option(state_arr[i],state_arr[i]);\n\t}\n}\n\nfunction print_city(city_id, city_index){\n\tvar option_str = document.getElementById(city_id);\n\toption_str.length=0;\t// Fixed by Julian Woods\n\toption_str.options[0] = new Option(\'Select City\',\'\');\n\toption_str.selectedIndex = 0;\n\tvar city_arr = s_a[city_index].split("|");\n\tfor (var i=0; i<city_arr.length; i++) {\n\t\toption_str.options[option_str.length] = new Option(city_arr[i],city_arr[i]);\n\t}\n}\n