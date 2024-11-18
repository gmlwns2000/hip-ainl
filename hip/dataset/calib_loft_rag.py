import json

prefix = r"""
@@@@@ CHUNK (user, text/plain) @@@@@
You are a helpful assistant.

@@@@@ CHUNK (user, text/plain) @@@@@

Your final answer should be a list of answers, in the following format:
Final Answer: ['answer1', 'answer2', ...]
If there is only one answer, it should be in the format:
Final Answer: ['answer1']

If there is no perfect answer output the closest one. Do not give an empty final answer.

-------------------------------------------------------------------------------
- From now, I will provide relavant documents. Please read carefully. Feel free to reference the document during answering question.
-------------------------------------------------------------------------------

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 0 | TITLE: English compound | CONTENT: Major style guides advise consulting a dictionary to determine whether a compound modifier should be hyphenated; the dictionary's hyphenation should be followed even when the compound modifier follows a noun (that is, regardless of whether in attributive or predicative position), because they are permanent compounds[5][6] (whereas the general rule with temporary compounds is that hyphens are omitted in the predicative position because they are used only when necessary to prevent misreading, which is usually only in the attributive position, and even there, only on a case-by-case basis).[7][8] | END ID: 0

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 1 | TITLE: The Lord of the Rings: The Return of the King | CONTENT: The music was composed by Howard Shore, who previously composed the first two parts of the trilogy. Shore watched the assembly cut of the film,[34] and had to write seven minutes of music per day to keep up with the schedule.[40] The score sees the full introduction of the Gondor theme, originally heard during Boromir's speeches at the Council of Elrond in The Fellowship of the Ring and at Osgiliath in The Two Towers' Extended Edition. Shore also used the Gondor theme with the new ascending coda (which is unique to this film) in his score for the trailer of the film. | END ID: 1

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 2 | TITLE: A Chinese Odyssey | CONTENT: A third film, A Chinese Odyssey Part Three, was released in China on September 14, 2016.[2] | END ID: 2

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 3 | TITLE: Interstate 476 | CONTENT: Originally planned as far back as 1929, the Mid-County Expressway was later proposed by the Pennsylvania Turnpike Commission as the "Chester Extension" of the Pennsylvania Turnpike in 1954. After the advent of the Interstate Highway System, the project was transferred to the Pennsylvania Department of Highways to be built as part of the system, designating it first as Interstate 495, and later as Interstate 480, as I-76 was designated as I-80S at the time. The present-day I-476 designation was assigned on February 6, 1964, when I-80S was renumbered as I-76.[42] | END ID: 3

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 4 | TITLE: School uniforms by country | CONTENT: Nowadays, many pre-schools advise parents to dress their children with a grembiulino, i.e., a small grembiule, usually shorter and more colourful, that can be purchased cheaply. | END ID: 4

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 5 | TITLE: Pinta Island tortoise | CONTENT: "Lonesome George" along with other of the tortoises on Pinta Island, belong to a genus of 21 species. Giant tortoises were widespread on most of the continents except for Australia and Antarctica. Not only do the Galapagos tortoises remain the largest living tortoises, but in the Galapagos, distinct populations survived in multiple localities. | END ID: 5

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 6 | TITLE: Middle kingdoms of India | CONTENT: During 9th-12th century, the Tomaras of Delhi ruled parts of the present-day Delhi and Haryana.[61] Much of the information about this dynasty comes from bardic legends of little historical value, and therefore, the reconstruction of their history is difficult.[62] According to the bardic tradition, the dynasty's founder Anangapal Tuar (that is Anangapala I Tomara) founded Delhi in 736 CE.[63] However, the authenticity of this claim is doubtful.[62] The bardic legends also state that the last Tomara king (also named Anangapal) passed on the throne of Delhi to his maternal grandson Prithviraj Chauhan. This claim is also inaccurate: historical evidence shows that Prithviraj inherited Delhi from his father Someshvara.[62] According to the Bijolia inscription of Someshvara, his brother Vigraharaja IV had captured Dhillika (Delhi) and Ashika (Hansi); he probably defeated a Tomara ruler.[64] | END ID: 6

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 7 | TITLE: Judith Keppel | CONTENT: Judith Cynthia Aline Keppel (born 18 August 1942)[2] was the first one-million-pound winner on the television game show Who Wants to Be a Millionaire? in the United Kingdom. She is also the only woman in the United Kingdom to have won it and also the first person to win a million pounds or more on a British television game show. She has appeared on the BBC Two quiz show Eggheads since 2003. | END ID: 7

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 8 | TITLE: Playing It My Way | CONTENT: Playing It My Way is the autobiography of former Indian cricketer Sachin Tendulkar. It was launched on 5 November 2014 in Mumbai.[3][4][5] The book summarises Tendulkar's early days, his 24 years of international career and aspects of his life that have not been shared publicly.[6] It entered the Limca Book of Records for being the best selling adult hardback across both fiction and non-fiction categories. In India, it broke the record set by Walter Isaacson's biography of Steve Jobs for being the most pre-ordered biographical book.[7] | END ID: 8

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 9 | TITLE: Jersey City, New Jersey | CONTENT: Jersey City is one of the most ethnically diverse cities in the world.[116][117] The city is a major port of entry for immigration to the United States and a major employment center at the approximate core of the New York City metropolitan region; and given its proximity to Manhattan, Jersey City has evolved a globally cosmopolitan ambiance of its own, demonstrating a robust and growing demographic and cultural diversity with respect to metrics including nationality, religion, race, and domiciliary partnership.[116] | END ID: 9

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 10 | TITLE: Great Depression in India | CONTENT: When the war came to an end, the Montagu-Chelmsford reforms were enacted in order to provide certain concessions to Indians in return for their loyalty to the Empire during the war. In 1923, the British Raj offered government protection to nine industries posing them as a sincere bid to industrialize the economy.[9] However, the measures appeared symbolic and were intended to finance and protect British enterprise as was evident from the fact that all the benefactors were British-run industries.[9] At the onset of the Great Depression, as it had been always, much of India's imports were from the United Kingdom.[9] On the eve of the First World War, India was the British Empire's single largest market with its exports to India at Rs. 730 million making up over one-sixth of the country's total exports.[10] | END ID: 10

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 11 | TITLE: Spanish Texas | CONTENT: The new missions were over 400 miles (640 km) from the nearest Spanish settlement, San Juan Bautista.[28] It was difficult to reprovision the missions, and by 1718 the missionaries were in dire straits.[31] Martín de Alarcón, who had been appointed governor of Texas in late 1716, wished to establish a way station between the settlements along the Rio Grande and the new missions in East Texas. The Coahuiltecans had built a thriving community near the headwaters of the San Antonio River,[32] in the area the Spanish had admired in 1707. Alarcón led a group of 72 people, including 10 families, into Texas on April 9, 1718. They brought with them 548 horses, 6 droves of mules, and other livestock. On May 1, the group created a temporary mud, brush and straw structure to serve as a mission, San Antonio de Valero, whose chapel was later known as the Alamo. The mission was initially populated with three to five Indians that one of the missionaries had raised since childhood. Alarcon built a presidio, San Antonio de Béxar one mile (1.6 km) north of the mission,.[33] Alarcón also chartered the municipality of Béjar, now San Antonio. Given a status higher than a village (pueblo) but lower than a city (ciudad), San Antonio became the only villa in Texas, and the colonists who settled there relied on farming and ranching to survive.[32] With the new settlement established, Alarcón continued on to the East Texas missions, where he found evidence of much illicit trade with France.[34] | END ID: 11

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 12 | TITLE: Marcelo H. del Pilar | CONTENT: In 1878, del Pilar resumed his law studies at the UST.[24] He earned his licenciado en jurisprudencia (equivalent to a Bachelor of Laws) in 1880.[26] After finishing law, he worked for the Real Audiencia de Manila (Royal Audience of Manila). | END ID: 12

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 13 | TITLE: Alt key | CONTENT: The Alt key has come to replace the Meta key of the old MIT keyboards. In their original function, both Alt and Meta would set the high bit of the signal generated by the key to 1 (for example, A generates 01000001 while Alt+A generates 11000001). However, in modern software, due to the requirement of the high bit for internationalization, Alt no longer works in such a way. | END ID: 13

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 14 | TITLE: San Cristóbal de las Casas | CONTENT: Casa Na Bolom (House of the Jaguar) is a museum, hotel and restaurant located outside the city’s historic center. The structure was built as part of a seminary in 1891, but it became the home of Frans Blom and Gertrude Duby Blom in the 20th century. Franz was an explorer and archeologist and Gertrude was a journalist and photographer. The couple spent over fifty years in Chiapas collecting tools, crafts, archeological pieces and clothing, especially related to the Lacandon Jungle and people. The museum is dedicated to this collection along with keeping some of the old household rooms intact, such as Franz’s study.[2][4] It also contains a library with more than 10,000 volumes dedicated to the history, culture and anthropology of the region. There are also magazine and sound libraries as well as the old chapel which contains colonial era religious art. The back of the structure contains a botanical garden.[1] | END ID: 14

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 15 | TITLE: Rabies vaccine | CONTENT: Imrab is an example of a veterinary rabies vaccine containing the Pasteur strain of killed rabies virus. Several different types of Imrab exist, including Imrab, Imrab 3, and Imrab Large Animal. Imrab 3 has been approved for ferrets and, in some areas, pet skunks.[19][20] | END ID: 15

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 16 | TITLE: The Pilgrim's Progress | CONTENT: Scholars have pointed out that Bunyan may have been influenced in the creation of places in The Pilgrim's Progress by his own surrounding environment. Albert Foster[18] describes the natural features of Bedfordshire that apparently turn up in The Pilgrim's Progress. Vera Brittain in her thoroughly researched biography of Bunyan,[19] identifies seven locations that appear in the allegory. Other connections are suggested in books not directly associated with either John Bunyan or The Pilgrim's Progress.[citation needed] | END ID: 16

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 17 | TITLE: Pogue | CONTENT: "Pogey bait" is a reference to sweets or candy, which was in usage in the military as early as 1918. The term alludes to food (and other luxuries) rarely afforded to grunts in the field.  To an infantry soldier, the term "pogey bait", when used, in the possessive sense (i.e. "my pogey bait", "his pogey bait", etc.) refers to a personally acquired (not issued) stash of snacks and food.  Common items found in a bag of "pogey bait" include Ramen Noodles, hard candies (i.e. Werther's Originals, Jolly Ranchers, Dum Dums, etc.), Beef Jerky, Easy Cheese, and Vienna Sausages (among other things).  "Pogey bait" was/is used "in the field" not only as snacks and meal supplements, but also for bartering (commonly either for other food or for tobacco products).[7]  "Pogey-bait run" was used as early as the 1960s to refer to any unauthorized violation of restrictions with the purpose of meeting a wife or girlfriend.[8] | END ID: 17

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 18 | TITLE: DualShock | CONTENT: The DualShock controller was widely supported; shortly after its launch most new titles, including Crash Bandicoot: Warped, Spyro the Dragon, and Tekken 3 included support for the vibration feature and dual analog sticks, while Capcom re-released Resident Evil: Director's Cut and Resident Evil 2 with support for the controller added to these newer versions. Some games designed for the Dual Analog's vibration capability, such as Porsche Challenge and Crash Bandicoot 2, also work. Many games took advantage of the presence of two motors to provide vibration effects in stereo including Gran Turismo and the PlayStation port of Quake II. Released in 1999, the PlayStation hit Ape Escape became the first game to explicitly require DualShock/Dual-Analog-type controllers, with its gameplay requiring the use of both analog sticks. In 2000, when the PS one (a remodeled version of the original PlayStation) was released with the slightly redesigned DualShock Controller (SCPH-110), similar to the first one, except its color is white instead of gray, in the middle of the controller has the "PS one" logo, instead of the "PlayStation" naming, most of the buttons, analog sticks and the cord are brighter than the previous one, and the connector is more of a semi-circle shape than having round edge, it also came in colors. | END ID: 18

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 19 | TITLE: Chickenpox | CONTENT: The diagnosis of chickenpox is primarily based on the signs and symptoms, with typical early symptoms followed by a characteristic rash. Confirmation of the diagnosis is by examination of the fluid within the vesicles of the rash, or by testing blood for evidence of an acute immunologic response. | END ID: 19

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 20 | TITLE: Continent | CONTENT: Islands are frequently grouped with a neighbouring continent to divide all the world's land into geopolitical regions. Under this scheme, most of the island countries and territories in the Pacific Ocean are grouped together with the continent of Australia to form a geopolitical region called Oceania. | END ID: 20

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 21 | TITLE: Orange (colour) | CONTENT: Flag of the Orange Order, an international Protestant fraternal organisation | END ID: 21

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 22 | TITLE: Television licensing in the United Kingdom | CONTENT: TV Licensing offers the following advice to those who have a TV but 'who wish to make it clear that they do not need a licence':[69] | END ID: 22

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 23 | TITLE: So (album) | CONTENT: The studio's basic equipment consisted of "two analog 24-track machines, a Studer A80, and a Studer A80 shell that had been modified by a local electronics wizard, with its own audio cards and transport controls".[nb 2] To record vocals a Neumann U47 tube microphone and a Decca compressor were used without equalization.[16] All of So's songs were made in a similar format. Gabriel would record a piano demo on a modified "B machine" and play this to the band. During rehearsals, the band would listen to the B machine through headphones and record their output onto the "A machine"; parts of Gabriel's demo would also be transferred to the A machine at this stage. Subsequent takes of the song were then put onto the B machine in order for the band to hear what they had played with the demo, as well as the song's new and old takes.[16] | END ID: 23

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 24 | TITLE: William Harvey | CONTENT: In 1628 he published in Frankfurt his completed treatise on the circulation of the blood, the De Motu Cordis. As a result of negative comments by other physicians Harvey "fell mightily in his practice",[15] but continued advancing his career. He was re-elected 'Censor' of the College of Physicians in 1629, having been elected for the first time in 1613 and the second time in 1625. Eventually, Harvey was also elected Treasurer of the College. | END ID: 24

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 25 | TITLE: Printing | CONTENT: Movable type is the system of printing and typography using movable pieces of metal type, made by casting from matrices struck by letterpunches. Movable type allowed for much more flexible processes than hand copying or block printing. | END ID: 25

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 26 | TITLE: Hyderabad State (1948–56) | CONTENT: Hyderabad State was a state in Independent India, formed after the accession of the princely state of Hyderabad into the Indian Union on 24 November 1949. It existed from 1948 to 1956. | END ID: 26

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 27 | TITLE: Interface (Java) | CONTENT: Another use of interfaces is being able to use an object without knowing its type of class, but rather only that it implements a certain interface. For instance, if one were annoyed by a whistling noise, one may not know whether it is a human or a parrot, because all that could be determined is that a whistler is whistling. The call whistler.whistle() will call the implemented method whistle of object whistler no matter what class it has, provided it implements Whistler. In a more practical example, a sorting algorithm may expect an object of type Comparable. Thus, without knowing the specific type, it knows that objects of that type can somehow be sorted. | END ID: 27

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 28 | TITLE: Dancing on Ice | CONTENT: Phillip Schofield and Christine Bleakley returned to co-present. Dean, Torvill and Karen Barber returned to mentor the celebrities. Robin Cousins, Jason Gardiner, Barber and Ashley Roberts returned for their respective ninth, eighth, seventh and second series on The Ice Panel. Cousins was absent for weeks 6 and 7 due to commentating the 2014 Winter Olympics, so former judge Nicky Slater returned in his place and Barber was temporary head judge. | END ID: 28

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 29 | TITLE: Cristiano Ronaldo | CONTENT: Born and raised on the Portuguese island of Madeira, Ronaldo was diagnosed with a racing heart at age 15. He underwent an operation to treat his condition, and began his senior club career playing for Sporting CP, before signing with Manchester United at age 18 in 2003. After winning his first trophy, the FA Cup, during his first season in England, he helped United win three successive Premier League titles, a UEFA Champions League title, and a FIFA Club World Cup. By age 22, he had received Ballon d'Or and FIFA World Player of the Year nominations and at age 23, he won his first Ballon d'Or and FIFA World Player of the Year awards. In 2009, Ronaldo was the subject of the most expensive association football transfer[note 3] when he moved from Manchester United to Real Madrid in a transfer worth €94 million (£80 million). | END ID: 29

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 30 | TITLE: List of awards and nominations received by Taylor Swift | CONTENT: Neox Fan Awards | END ID: 30

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 31 | TITLE: Fortunate Son | CONTENT: It attracted criticism when Bruce Springsteen, Dave Grohl, and Zac Brown performed the song together at the November 2014 Concert for Valor in Washington D.C.. Fogerty, a military veteran, defended their song choice.[12] | END ID: 31

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 32 | TITLE: Shambala (song) | CONTENT: The well-known cover of this song by the rock band Three Dog Night appeared in 1973 on the Billboard Hot 100, on the top 40 from the beginning of June through the end of August, reaching #3 in both the pop singles and adult contemporary categories,[1] #1 on the Cashbox Magazine charts,[2] and an isolated week at #1 on WLS.[3] Headed toward the Hot 100's summit in late July, had it not run out of steam, “Shambala” would have completed an uncommon distinction of a Hot 100 chart-topper for each of four consecutive years for the group.  The song, the first one that the group had specifically cut as a single, rather than an album cut,[4] later appeared on Cyan, Three Dog Night's ninth album, and subsequently on numerous anthologies and compilation albums.[1][5] | END ID: 32

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 33 | TITLE: Memento (film) | CONTENT: David Julyan composed the film's synthesized score. Julyan acknowledges several synthesized soundtracks that inspired him, such as Vangelis's Blade Runner and Hans Zimmer's The Thin Red Line.[32] While composing the score, Julyan created different, distinct sounds to differentiate between the color and black-and-white scenes: "brooding and classical" themes in the former, and "oppressive and rumbly noise" in the latter.[33] Since he describes the entire score as "Leonard's theme", Julyan says, "The emotion I was aiming at with my music was yearning and loss. But a sense of loss you feel but at the same time you don't know what it is you have lost, a sense of being adrift."[34] Initially, Nolan wanted to use Radiohead's "Paranoid Android" during the end credits, but he was unable to secure the rights.[35] Instead, David Bowie's "Something in the Air" is used, although another of Radiohead's songs, an extended version of "Treefingers", is included on the film's soundtrack.[36] | END ID: 33

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 34 | TITLE: Jon Bon Jovi | CONTENT: In late 2013, it was rumored that Jon Bon Jovi would enter the race to bid for the Buffalo Bills following the death of long-time owner Ralph Wilson. Bon Jovi denied the rumors. However, in June 2014, it was confirmed that he along with a sports ownership group from Toronto were intending to bid on the team. Bon Jovi and his ownership group made it to the final round of bidding, but the team was sold to Buffalo Sabres owner Terry Pegula.[39] | END ID: 34

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 35 | TITLE: The Fab Four (tribute) | CONTENT: The original group, which includes McNeil, along with Ardy Sarraf, Rolo Sandoval and Michael Amador, have performed together as The Fab Four for the past 12 years, covering nearly the entire Beatles songbook, plus solo material as well. | END ID: 35

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 36 | TITLE: University of the Virgin Islands | CONTENT: The university offers counseling and career services including: interpersonal, personal, social and cognitive development education. | END ID: 36

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 37 | TITLE: Desalination | CONTENT: The Reverse Osmosis process is not maintenance free. Various factors interfere with efficiency: ionic contamination (calcium, magnesium etc.); DOC; bacteria; viruses; colloids & insoluble particulates; biofouling and scaling. In extreme cases the RO membranes are destroyed. To mitigate damage, various pretreatment stages are introduced. Anti-scaling inhibitors include acids and other agents like the organic polymers Polyacrylamide and Polymaleic Acid), Phosphonates and Polyphosphates. Inhibitors for fouling are biocides (as oxidants against bacteria and viruses), like chlorine, ozone, sodium or calcium hypochlorite. At regular intervals, depending on the membrane contamination; fluctuating seawater conditions; or when prompted by monitoring processes, the membranes need to be cleaned, known as emergency or shock-flushing. Flushing is done with inhibitors in a fresh water solution and the system must go offline. This procedure is environmental risky, since contaminated water is diverted into the ocean without treatment. Sensitive marine habitats can be irreversibly damaged.[13][14] | END ID: 37

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 38 | TITLE: Monel | CONTENT: Monel has also been used in Kelvinator refrigerators. | END ID: 38

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 39 | TITLE: Non-ferrous metal | CONTENT: Generally more expensive than ferrous metals, non-ferrous metals are used because of desirable properties such as low weight (e.g. aluminium), higher conductivity (e.g. copper),[1] non-magnetic property or resistance to corrosion (e.g. zinc).[2] Some non-ferrous materials are also used in the iron and steel industries. For example, bauxite is used as flux for blast furnaces, while others such as wolframite, pyrolusite and chromite are used in making ferrous alloys.[3] | END ID: 39

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 40 | TITLE: List of multiple Winter Olympic medalists | CONTENT: This list shows only the athletes who have won at least eight medals at the Winter Olympics. | END ID: 40

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 41 | TITLE: Clearing house (finance) | CONTENT: The Americans improved on the British check clearing system and opened a bankers' clearing house in the Bank of New York on Wall Street, New York in 1853. Instead of the slow London procedure in which each bank clerk, one at a time, stepped up to an Inspector's rostrum, in the New York procedure two bank clerks from each bank all worked simultaneously. One clerk from each bank sat inside a 70 foot long oval table, while the second clerk from each bank stood outside the table facing the other clerk from the same bank.[7] Each of the outside clerks carried a file box. When the manager signaled, all of the outside clerks stepped one position to the left, to face the next seated clerks. If a seated clerk represented a bank to which money was owed or from which money was receivable, the net amount of cash would change hands, along with checks and paper documents. | END ID: 41

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 42 | TITLE: Artificial intelligence | CONTENT: With social media sites overtaking TV as a source for news for young people and news organisations increasingly reliant on social media platforms for generating distribution,[246] major publishers now use artificial intelligence (AI) technology to post stories more effectively and generate higher volumes of traffic.[247] | END ID: 42

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 43 | TITLE: List of Peep Show episodes | CONTENT: The two men are competing to try to become their unhappily married next-door neighbour Toni's fuck buddy. We follow them to a party at Toni's, at which Mark fails in his attempt to seduce her while talking to her about the Battle of Stalingrad. Meanwhile, Jez meets her sister â€“ whom he erroneously believes has leukaemia because she has another sister who has the disease. After this misunderstanding causes an argument, Mark and Jez are told by Toni's husband, Tony, to leave. | END ID: 43

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 44 | TITLE: List of My Babysitter's a Vampire episodes | CONTENT: Vanessa Morgan confirmed via Twitter that there will be a second season.[7] It was later confirmed and that it consisted of 13 episodes.[8] Season 2 began filming on September 21, 2011 and wrapped up on November 15, 2011. Season 2 was produced by Byron A. Martin who also acted as 2nd Unit Director on numerous episodes. Disney Channel announced they would pick it up again for a second season.[9] | END ID: 44

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 45 | TITLE: Edge (wrestler) | CONTENT: Copeland was trained by professional wrestlers Sweet Daddy Siki and Ron Hutchison. Throughout the 1990s, he wrestled in various United States independent promotions. During his time in these promotions, he competed in singles and tag team competition, the latter with long-time friend Christian. In 1997, Copeland signed a developmental deal with the WWF and began competing for the company later that year; he made his televised debut the following June under the ring name Edge. In July 1999, he won the WWF Intercontinental Championship at a house show in Toronto, making it his first title reign with the company. Edge and Christian, billed as brothers and later childhood friends in WWF/WWE storylines, went on to win the WWF Tag Team Championship on seven different occasions. During this time, they gained notoriety in the tag team division, partly due to their participation in Tables, Ladders, and Chairs matches. | END ID: 45

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 46 | TITLE: Arkansas Razorbacks football | CONTENT: On December 4, 2012, it was announced that Bret Bielema would leave the Wisconsin Badgers to become the head coach of the Razorbacks for the 2013 season.[72][73] Bielema is the 33rd head football coach in Arkansas history. | END ID: 46

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 47 | TITLE: List of ships of the line of the Royal Navy | CONTENT: The 1706 Establishment established a desired set of principal dimensions for each group (i.e. size) of warship from the 40-gun fifth rate up to the 90-gun second rate (first rates and ships of less than 40 guns were not covered by the 1706 Establishment). As only the principal dimensions were specified, the design of individual ships remained with the Master Shipwright in each Dockyard; thus ships of the same number of guns built to this Establishment did not constitute a class in the modern sense of all being built to one design. | END ID: 47

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 48 | TITLE: Heath Ledger | CONTENT: Ledger died on 22 January 2008[5][1] from an accidental intoxication from prescription drugs.[7][8][9] A few months before his death, Ledger had finished filming his performance as The Joker in The Dark Knight. His death occurred during editing of The Dark Knight and in the midst of filming his last role as Tony in The Imaginarium of Doctor Parnassus. His untimely death cast a shadow over the subsequent promotion of the $185 million Batman production.[10] Ledger received numerous posthumous accolades for his critically acclaimed performance in the film, including the Academy Award for Best Supporting Actor, a Best Actor International Award at the 2008 Australian Film Institute Awards (for which he became the first actor to win an award posthumously),[11] the 2008 Los Angeles Film Critics Association Award for Best Supporting Actor, the 2009 Golden Globe Award for Best Supporting Actor – Motion Picture,[12] and the 2009 BAFTA Award for Best Supporting Actor.[4] | END ID: 48

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 49 | TITLE: Candlestick Park | CONTENT: In addition to Clark's famous touchdown catch, two more plays referred to as "The Catch" took place during games at Candlestick. The play dubbed "The Catch II" came in the 1998 Wild Card round, as Steve Young found Terrell Owens for a touchdown with eight seconds left to defeat the two-time defending NFC Champion Packers. The play called "The Catch III" came in the 2011 Divisional Playoffs, when Alex Smith threw a touchdown pass to Vernon Davis with nine seconds remaining to provide the winning margin against the New Orleans Saints. | END ID: 49

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 50 | TITLE: Mongolia | CONTENT: Freestyle wrestling has been practised since 1958 in Mongolia.[123] Mongolian freestyle wrestlers have won the first and the most Olympic medals of Mongolia. | END ID: 50

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 51 | TITLE: Mercator projection | CONTENT: One measure of a map's accuracy is a comparison of the length of corresponding line elements on the map and globe. Therefore, by construction, the Mercator projection is perfectly accurate, k = 1, along the equator and nowhere else. At a latitude of ±25° the value of sec φ is about 1.1 and therefore the projection may be deemed accurate to within 10% in a strip of width 50° centred on the equator. Narrower strips are better: sec 8° = 1.01, so a strip of width 16° (centred on the equator) is accurate to within 1% or 1 part in 100. Similarly sec 2.56° = 1.001, so a strip of width 5.12° (centred on the equator) is accurate to within 0.1% or 1 part in 1,000. Therefore, the Mercator projection is adequate for mapping countries close to the equator. | END ID: 51

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 52 | TITLE: Civil procedure in South Africa | CONTENT: Rule 42(1) supplements the common law by providing for certain instances in which the court may, either mero motu or on application by one of the parties, set aside or vary one of its judgments or orders. The element that is more or less common to all the instances of variation or rescission under this rule is that of error. The rule provides for variation in the following instances: | END ID: 52

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 53 | TITLE: Climate of Georgia (U.S. state) | CONTENT: Most of Georgia has a sub-tropical climate, with hot and humid summers, except at the highest elevations. Weather conditions in various localities of Georgia depend on how close they are to the Atlantic Ocean or Gulf of Mexico, and their altitude. This is especially true in the mountainous areas in the northern part of the state, which are farther away from ocean waters and can be up to 4,500 feet (1,400Â m) or higher above sea level. The areas near the Florida-Georgia border, extending from the Atlantic Ocean westward to the Chattahoochee River, experience the most subtropical weather, similar to that of Florida: hot, humid summers with frequent afternoon thunderstorms and mild, somewhat drier winters. | END ID: 53

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 54 | TITLE: I've Got a Tiger By the Tail | CONTENT: Owens — in the liner notes to The Buck Owens Collection: 1959-1990 — recalled that he and songwriter Harlan Howard had gotten together to write songs, but things were going slowly. Then, Owens saw an Esso gas station sign with the company's slogan at the time, "Put a tiger in your tank" ... and got an idea.[1] | END ID: 54

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 55 | TITLE: 2018 Commonwealth Games opening ceremony | CONTENT: The Courier-Mail said "Gold Coast finally welcomed the world to its biggest ever party with a dazzling Commonwealth Games opening ceremony".[44] The New Daily said the opening ceremony had "wowed" the fans on the Gold Coast.[45] The Gold Coast Bulletin called the opening ceremony as "dazzling" and "welcomed the world to its biggest ever party".[46] The West Australian said that the "Spirits gone high" in the Gold Coast after the opening ceremony.[47] The SBS called the ceremony as "dazzling"[48] and the ABC said that the opening ceremony had "signaled great start for the 2018 Commonwealth Games".[49] | END ID: 55

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 56 | TITLE: Kyle Craig | CONTENT: Kyle Craig is mentioned at the beginning of the novel and appears at the very end. Alex Cross comes to him for help, which Craig provides, and gives the name of the criminal mastermind, The Wolf, but the target is revealed to be a decoy. | END ID: 56

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 57 | TITLE: Birds of Australia | CONTENT: Australian king parrot | END ID: 57

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 58 | TITLE: Infertility | CONTENT: The main cause of male infertility is low semen quality. In men who have the necessary reproductive organs to procreate, infertility can be caused by low sperm count due to endocrine problems, drugs, radiation, or infection. There may be testicular malformations, hormone imbalance, or blockage of the man's duct system. Although many of these can be treated through surgery or hormonal substitutions, some may be indefinite.[57] Infertility associated with viable, but immotile sperm may be caused by primary ciliary dyskinesia. The sperm must provide the zygote with DNA, centrioles, and activation factor for the embryo to develop. A defect in any of these sperm structures may result in infertility that will not be detected by semen analysis.[58] Antisperm antibodies cause immune infertility.[22][23] Cystic fibrosis can lead to infertility in men. | END ID: 58

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 59 | TITLE: Academy Awards | CONTENT: The Academy Awards have long received criticism over its lack of diversity among the nominees.[83][84][85] This criticism is based on the statistics from every Academy Awards since 1929 which shows us that only 6.4% of academy award nominees have been non-white and since 1991, 11.2% of nominees have been non-white, with the rate of winners being even more polarizing [86]. The 88th awards ceremony became the target of a boycott, popularized on social media by the #OscarsSoWhite, based on critics' perception that its all-white acting nominee list reflected bias. In response, the Academy initiated "historic" changes in membership by the year 2020.[87][88] | END ID: 59

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 60 | TITLE: Mandala | CONTENT: Painted 19th century Tibetan mandala of the Naropa tradition, Vajrayogini stands in the center of two crossed red triangles, Rubin Museum of Art | END ID: 60

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 61 | TITLE: Routing | CONTENT: When one network node goes down, any nodes that used it as their next hop discard the entry, and create new routing-table information. These nodes convey the updated routing information to all adjacent nodes, which in turn repeat the process. Eventually all the nodes in the network receive the updates, and discover new paths to all the destinations they can still "reach". | END ID: 61

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 62 | TITLE: Judicial activism | CONTENT: All such rulings carry the force of Article 39A of the Constitution of India,[25] although before and during the Emergency the judiciary desisted from "wide and elastic" interpretations, termed Austinian, because Directive Principles of State Policy are non-justiciable. This despite the constitutional provisions for judicial review and B R Ambedkar arguing in the Constituent Assembly Debates that "judicial review, particularly writ jurisdiction, could provide quick relief against abridgment of Fundamental Rights and ought to be at the heart of the Constitution."[26] | END ID: 62

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 63 | TITLE: List of Eurovision Song Contest winners | CONTENT: There have been 62 contests, with one winner each year except the tied 1969 contest, which had four. Twenty-seven different countries have won the contest. Switzerland won the first contest in 1956. The country with the highest number of wins is Ireland, with seven. The only person to have won more than once as performer is Ireland's Johnny Logan, who performed "What's Another Year" in 1980 and "Hold Me Now" in 1987. Logan is also one of only five songwriters to have written more than one winning entry ("Hold Me Now" 1987 and "Why Me?" 1992, performed by Linda Martin).[3] This unique distinction makes Logan the only person to have three Eurovision victories to his/her credit, as either singer, songwriter or both. The other four songwriters with more than one winning entry to their credit are, Willy van Hemert (Netherlands, 1957 and 1959), Yves Dessca (Monaco, 1971 and Luxembourg, 1972), Rolf Løvland (Norway, 1985 and 1995) and Brendan Graham (Ireland, 1994 and 1996). | END ID: 63

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 64 | TITLE: Eligible receiver | CONTENT: In both American and Canadian professional football, every player on the defensive team is considered eligible. The offensive team must have at least seven players lined up on the line of scrimmage. Of the players on the line of scrimmage, only the two players on the ends of the line of scrimmage are eligible receivers. The remaining players are in the backfield (four in American football, five in Canadian football), including the quarterback. These backfield players are also eligible receivers. In the National Football League, a quarterback who takes his stance behind center as a T-formation quarterback is not eligible unless, before the ball is snapped, he legally moves to a position at least one yard behind the line of scrimmage or on the end of the line, and is stationary in that position for at least one second before the snap, but is nonetheless not counted toward the seven men required on the line of scrimmage.[3] | END ID: 64

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 65 | TITLE: De amore (Andreas Capellanus) | CONTENT: De Amore was written sometime between 1186 and 1190. It was most likely intended for the French court of Philip Augustus. It has been supposed to have been written in 1185 at the request of Marie de Champagne, daughter of King Louis VII of France and of Eleanor of Aquitaine.[1] A dismissive allusion in the text to the "wealth of Hungary" has suggested the hypothesis that it was written after 1184, at the time when Bela III of Hungary had sent to the French court a statement of his income and had proposed marriage to Marie's sister Marguerite of France, but before 1186, when his proposal was accepted. | END ID: 65

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 66 | TITLE: Cuban Missile Crisis | CONTENT: BBC journalist Joe Matthews published the story, on October 13, 2012, behind the 100 tactical nuclear warheads mentioned by Graham Allison in the excerpt above.[124] Khrushchev feared that Castro's hurt pride and widespread Cuban indignation over the concessions he had made to Kennedy might lead to a breakdown of the agreement between the Soviet Union and the US. To prevent that, Khrushchev decided to offer to give Cuba more than 100 tactical nuclear weapons that had been shipped to Cuba along with the long-range missiles but, crucially, had escaped the notice of U.S. intelligence. Khrushchev determined that because the Americans had not listed the missiles on their list of demands, keeping them in Cuba would be in the Soviet Union's interests.[124] | END ID: 66

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 67 | TITLE: United States Attorney | CONTENT: The Office of the United States Attorney was created by the Judiciary Act of 1789, along with the office of Attorney General and the United States Marshals Service. The same act also specified the structure of the Supreme Court of the United States and established inferior courts making up the United States Federal Judiciary,  including  a district court system.  Thus, the office of U.S. Attorney  is older than the Department of Justice.  The Judiciary Act of 1789  provided for the appointment in each judicial district of a "Person learned in the law to act as attorney for the United States...whose duty it shall be to prosecute in each district all delinquents for crimes and offenses cognizable under the authority of the United States, and all civil actions in which the United States shall be concerned..."
Prior to the existence of the Department of Justice, the U.S. Attorneys were independent of the Attorney General, and did not come under the AG's supervision and authority until 1870, with the creation of the Department of Justice.[8][9] | END ID: 67

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 68 | TITLE: Salad Fingers | CONTENT: Salad Fingers is sitting in his armchair, trying to tune his radio which he calls "Roger." If he is lucky, Salad Fingers says he may chance upon a broadcast from "Croxley", which so happens to be a small town in Hertfordshire. "Croxleyheath" also occurs in Shore Leave. After feeding Roger his "sustenance" (which seems to be marbles, peas, rocks or beans), it begins to emit a strange, piercing frequency. A gurgling sound comes from Salad Fingers' own stomach, insinuating upset in reaction to the "unpleasant frequencies" coming from the radio. He decides to wait out the tormenting event in his "safety cupboard." | END ID: 68

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 69 | TITLE: Music of immigrant communities in the United States | CONTENT: See: Music of Jamaica | END ID: 69

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 70 | TITLE: Dehumidifier | CONTENT: Because window air conditioner units have condensers and expansion units, some of them can be used as makeshift dehumidifiers by sending their heat exhaust back into the same room as the cooled air, instead of the outside environment. If the condensate from the cooling coils is drained away from the room as it drips off the cooling coils, the result will be room air that is drier but slightly warmer. | END ID: 70

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 71 | TITLE: Agriculture | CONTENT: Cropping systems vary among farms depending on the available resources and constraints; geography and climate of the farm; government policy; economic, social and political pressures; and the philosophy and culture of the farmer.[106][107] | END ID: 71

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 72 | TITLE: Electrostatic discharge | CONTENT: A spark is triggered when the electric field strength exceeds approximately 4–30 kV/cm[2] — the dielectric field strength of air. This may cause a very rapid increase in the number of free electrons and ions in the air, temporarily causing the air to abruptly become an electrical conductor in a process called dielectric breakdown. | END ID: 72

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 73 | TITLE: Modern philosophy | CONTENT: 19th-century British philosophy came increasingly to be dominated by strands of neo-Hegelian thought, and as a reaction against this, figures such as Bertrand Russell and George Edward Moore began moving in the direction of analytic philosophy, which was essentially an updating of traditional empiricism to accommodate the new developments in logic of the German mathematician Gottlob Frege. | END ID: 73

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 74 | TITLE: Myanmar | CONTENT: Myanmar is known with a name deriving from Burma as opposed to Myanmar in Spanish, Italian, Romanian, and Greek – Birmania being the local version of Burma in the Spanish language, for example. Myanmar used to be known as "Birmânia" in Portuguese, and as "Birmanie" in French.[32] As in the past, French-language media today consistently use Birmanie.,[33][34] | END ID: 74

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 75 | TITLE: Judgement of Paris | CONTENT: In the Hercules: The Legendary Journeys series, the contest is altered somewhat with Aphrodite and Athena entering but Artemis is the third goddess contestant instead of Hera (offering the one who chooses her the chance to be renowned as a great warrior). The Golden Apple appears as a gift from Aphrodite with the ability to make any mortal woman fall in love with the man holding it and to make a mortal man and woman soul mates if they simultaneously touch it. The other major differences beside the presence of Artemis and the role of the apple are the fact that it is Iolaus who is the judge and the goddesses appear in swimsuits and not nude. | END ID: 75

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 76 | TITLE: Climate of Madrid | CONTENT: The highest temperature recorded during the day is 40.6 °C (105.1 °F) on the 10 August 2012. On the August 1933 reported record, the average maximum temperature during the day was 35.5 °C (95.9 °F). The coldest temperature recorded was −10.1 °C (13.8 °F) at night on 16 January 1945.[8] | END ID: 76

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 77 | TITLE: Michael Schumacher | CONTENT: 2006 became the last season of Schumacher's Ferrari career. After three races, Schumacher had just 11 points and was already 17 points behind Alonso. He won the following two races. His pole position at San Marino was his 66th, breaking Ayrton Senna's 12-year-old record.[91] | END ID: 77

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 78 | TITLE: Balaam | CONTENT: Balaam sends back word that he can only do what YHWH commands, and God has, via a nocturnal dream, told him not to go. Balak consequently sends higher-ranking priests and offers Balaam honours; Balaam continues to press God, and God finally permits him to go but with instructions to say only what he commands. Balaam then sets out in the morning with the princes of Moab. God becomes angry that he went, and sends the Angel of the Lord (Numbers 22:22) to prevent him. At first, the angel is seen only by the donkey Balaam is riding, which tries to avoid the angel. After Balaam starts punishing the donkey for refusing to move, it is miraculously given the power to speak to Balaam (Numbers 22:28), and it complains about Balaam's treatment. At this point, Balaam is allowed to see the angel, who informs him that the donkey is the only reason the angel did not kill Balaam. Balaam immediately repents, but is told to go on. | END ID: 78

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 79 | TITLE: The Waltons | CONTENT: The television movie The Homecoming: A Christmas Story was broadcast on December 19, 1971.[1] Based on its success, the CBS television network ordered one season of episodes based on the same characters and that became the television series The Waltons.[2] Beginning in September 1972, the series subsequently aired on CBS for nine seasons. After the series was canceled by CBS in 1981, NBC aired three television movie sequels in 1982, with three more in the 1990s on CBS. The Waltons was produced by Lorimar Productions and distributed by Warner Bros. Domestic Television Distribution in syndication. | END ID: 79

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 80 | TITLE: Dublin | CONTENT: Dublin Castle, which became the centre of Norman power in Ireland, was founded in 1204 as a major defensive work on the orders of King John of England.[31] Following the appointment of the first Lord Mayor of Dublin in 1229, the city expanded and had a population of 8,000 by the end of the 13th century. Dublin prospered as a trade centre, despite an attempt by King Robert I of Scotland to capture the city in 1317.[30] It remained a relatively small walled medieval town during the 14th century and was under constant threat from the surrounding native clans. In 1348, the Black Death, a lethal plague which had ravaged Europe, took hold in Dublin and killed thousands over the following decade.[32][33] | END ID: 80

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 81 | TITLE: The Land of Make Believe | CONTENT: "The Land of Make Believe" was the third single by the British band allSTARS*. The single was slightly faster than the original version and had a more euro-pop sound. The music video was set in a circus tent, with each individual member of the band performing tricks e.g. being cut in half, levitating or juggling. The single performed to moderate success, achieving allSTARS*' highest UK chart position of No.9.[26] | END ID: 81

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 82 | TITLE: Eastern gray squirrel | CONTENT: 20 different Pleistocene fauna specimens contain S. carolinensis, found in Florida and dated to be as early as the late Irvingtonian period.[13] Body size seems to have increased during the early to middle Holocene and then decreased to the present size seen today. | END ID: 82

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 83 | TITLE: Local government | CONTENT: Until 1996, the President appointed the mayor of Buenos Aires, and by law, the president and Congress controlled any legislation that affected the city. Constitutional reforms that year led to an elected mayoral position, and a 60-member Poder Legislativo (legislative power). | END ID: 83

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 84 | TITLE: The View (talk show) | CONTENT: Rosie O'Donnell's return to the show as a permanent co-host was announced on July 10, 2014.[92] Shortly after, the show conducted a "chemistry testing" of various prospective hosts to join Goldberg and O'Donnell.[24] On September 3, the series announced that Rosie Perez and Nicolle Wallace would join the panel as co-hosts for its eighteenth season.[93] Both Perez, an actress and choreographer, and Wallace, an MSNBC political analyst and former Bush White House communications chief, previously made guest appearances on The View during season 17.[94] On February 6, 2015, representatives for O'Donnell confirmed that she would once again exit the panel, citing her reasons as a "personal decision." Her final appearance aired on February 12.[95] On June 10, recurring guest panelist Raven-Symoné joined the series as a permanent co-host.[96] On July 7, it was announced that Perez would exit the series following the completion of its eighteenth season in order to fully pursue acting.[97] On July 15, it was announced that Wallace was being let go at the conclusion of season 18.[98] | END ID: 84

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 85 | TITLE: Slaughterhouse | CONTENT: By 2010 a mobile facility the Modular Harvest System had received USDA approval. It can be moved from ranch to ranch. It consists of three trailers, one for slaughtering, one for consumable body parts and one for other body parts. Preparation of individual cuts is done at a butchery or other meat preparation facility.[20] | END ID: 85

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 86 | TITLE: Transcendentalism | CONTENT: Transcendentalism is, in many aspects, the first notable American intellectual movement. It has inspired succeeding generations of American intellectuals, as well as some literary movements.[20] | END ID: 86

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 87 | TITLE: Dowry | CONTENT: Dowry was common in different historic periods of China and continued through the modern history. Locally called Jiàzhuāng (嫁妝), the dowry ranged from land, jewelry, money to a collection of clothing, sewing equipment and collection of household items. Mann[17] and others[47][48][49] find that dowry was a form of inheritance to daughters. In traditional China, the property owned by a family, if any, was earmarked for equal division or inheritance by sons only. Dowry was the only way assets were transferred to a daughter. It included immovable property such as land, and movable property like jewelry and fine clothing. The dowry she brought with her was typically sequestered from the property of her husband and other male members in a joint family. She would often sell this property for cash to overcome hard economic times or needs of her children and husband. In a few cases, she may transfer the property she brought as dowry to her daughter or daughter-in-law. Dowry assets once transferred in turn constituted separate wealth of the woman who received it (sifang qian, etc.). Often a woman who brought a large dowry was considered more virtuous in Chinese culture than one who didn't.[17] In parts of China, both dowry and brideprice (pinjin) were practiced from ancient eras to the 20th century. Though throughout the history of China, the practice of using a brideprice has largely been used instead of dowries, but has slowly diminished in modern times.[50] | END ID: 87

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 88 | TITLE: History of Egypt | CONTENT: On 28 April, another mass trial took place with 683 Morsi supporters sentenced to death for killing 1 police officer.[56] | END ID: 88

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 89 | TITLE: Maximus (comics) | CONTENT: In the alternate timeline seen in the 1995â€“1996 "Age of Apocalypse" storyline, Maximus was a Horseman of Apocalypse, the Horseman of Death. He operates on the Blue Area of the Moon, aboard Ship, Apocalypse's Celestial starship, whose sentient artificial intelligence is known as Ship. Maximus is served by his personal strikeforce formed by clones of the Inhuman Royal Family, which he had murdered himself, altered into monstrous forms by the Terrigen Mists, which Death has offered Apocalypse in exchange for his position. Maximus also experiments on Sunfire, who has been captured by Holocaust after the destruction of Japan, leaving him unable to control his powers. When the X-Men appear on the Moon, believing Apocalypse to be hibernating on Ship, Maximus capture the X-Men and seeks to transform them into his servants, with which he will overthrow Apocalypse. However, Cyclops, who has been sent to ensure the transfer of the Mists, attacks the betrayer Death and liberate the X-Men with the aid of Blink. Maximus dies, alongside his servants, in the destruction of Ship caused by Sunfire, whose powers flare out of control after he was released.[14] | END ID: 89

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 90 | TITLE: Taman Negara | CONTENT: From Kuala Lumpur, buses to Taman Negara National Park leave from Kompleks Selangor along Jalan Sultan in Petaling Street or Bangunan Mariamman, Jalan Hang Kasturi-nearby Pasar Seni Transportation Hub.(GoKLCityBus).Daily departure at 8.30am including public holiday. 3hrs traveling time from Kuala Lumpur to Kuala Tembeling Jetty and after that another 3hrs boat ride upstream to Kuala Tahan. | END ID: 90

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 91 | TITLE: Greek mythology | CONTENT: During the Hellenistic period, mythology took on the prestige of elite knowledge that marks its possessors as belonging to a certain class. At the same time, the skeptical turn of the Classical age became even more pronounced.[76] Greek mythographer Euhemerus established the tradition of seeking an actual historical basis for mythical beings and events.[77] Although his original work (Sacred Scriptures) is lost, much is known about it from what is recorded by Diodorus and Lactantius.[78] | END ID: 91

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 92 | TITLE: Marriage | CONTENT: The matrimonial covenant, by which a man and a woman establish between themselves a partnership of the whole of life, is by its nature ordered toward the good of the spouses and the procreation and education of offspring; this covenant between baptized persons has been raised by Christ the Lord to the dignity of a sacrament.[225] | END ID: 92

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 93 | TITLE: List of Olympic Games host cities | CONTENT: The Games have primarily been hosted in the continents of Europe (36 editions) and North America (12 editions); eight Games have been hosted in Asia and two have been hosted in Oceania. In 2016, Rio de Janeiro became South America's first Olympic host city, while the African continent has yet to hold the Games. Other major geographic regions which have never hosted the Olympics include the Middle East, the Indian subcontinent, the Caribbean, and Southeast Asia. | END ID: 93

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 94 | TITLE: Springfield (The Simpsons) | CONTENT: A buffet-style family restaurant, located in a grounded passenger plane. One of it's "features" is turbulence caused by staff rocking the plane back and forth by pushing on the wings. | END ID: 94

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 95 | TITLE: Necktie | CONTENT: In 1715, another kind of neckwear, called "stocks" made its appearance. The term originally referred to a leather collar, laced at the back, worn by soldiers to promote holding the head high in a military bearing.  The leather stock also afforded some protection to the major blood vessels of the neck from saber or bayonet attacks.  General Sherman is seen wearing a leather stock in several American Civil War-era photographs. | END ID: 95

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 96 | TITLE: Northern Ireland Assembly | CONTENT: A transferred matter is defined as "any matter which is not an excepted or reserved matter".[22] There is therefore no full listing of transferred matters but they have been grouped into the responsibilities of the Northern Ireland Executive ministers: | END ID: 96

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 97 | TITLE: List of early settlers of Rhode Island | CONTENT: The last four names on the list were crossed out, but these men nevertheless came to Portsmouth or Newport. | END ID: 97

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 98 | TITLE: Smallpox | CONTENT: U.S. Presidents George Washington, Andrew Jackson, and Abraham Lincoln all contracted and recovered from the disease. Washington became infected with smallpox on a visit to Barbados in 1751.[125] Jackson developed the illness after being taken prisoner by the British during the American Revolution, and though he recovered, his brother Robert did not.[125] Lincoln contracted the disease during his Presidency, possibly from his son Tad, and was quarantined shortly after giving the Gettysburg address in 1863.[125] | END ID: 98

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 99 | TITLE: 49th Academy Awards | CONTENT: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting, a feat matched only by fellow Australian Heath Ledger 32 years later; Finch had suffered a fatal heart attack in mid-January. Beatrice Straight set another record by becoming the actor with the shortest performance ever in a film to win an acting Oscar, with only five minutes and two seconds of screen-time in Network. Network, along with All the President's Men, were the two biggest champs of the ceremony with four Oscars each, but Best Picture and Best Director ultimately went to Rocky. | END ID: 99

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 100 | TITLE: Yugoslavia | CONTENT: Yugoslavia solved the national issue of nations and nationalities (national minorities) in a way that all nations and nationalities had the same rights. The flags of the republics used versions of the red flag or Slavic tricolor, with a red star in the centre or in the canton. | END ID: 100

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 101 | TITLE: Knights of Labor | CONTENT: As membership expanded, the Knights began to function more as a labor union and less of a secret organization. During the 1880's, the Knights of Labor played a huge role in independent and third-party movements.[7] Local assemblies began not only to emphasize cooperative enterprises, but to initiate strikes to win concessions from employers. The Knights of Labor brought together workers of different religion, race and gender and helped them all create a bond and unify all for the same cause.[8] The new leader Powderly, opposed strikes as a "relic of barbarism," but the size and the diversity of the Knights afforded local assemblies a great deal of autonomy. | END ID: 101

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 102 | TITLE: Old Glory | CONTENT: In order to save the flag from further threats, Driver (aided by loyal women neighbors) had it sewn into a coverlet and hidden until late February 1862, when Nashville fell to Union forces.[2] When the Union Army (led by the 6th Ohio Infantry) entered the city, Driver went to Tennessee State Capitol after seeing the American flag and the 6th Ohio's regimental colors raised on the Capitol flagstaff.[2] | END ID: 102

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 103 | TITLE: Ironclad warship | CONTENT: Ironclads were also used from the inception of the Imperial Japanese Navy. Kōtetsu (Japanese: 甲鉄, literally "Ironclad", later renamed Azuma 東, "East") had a decisive role in the Naval Battle of Hakodate Bay in May 1869, which marked the end of the Boshin War, and the complete establishment of the Meiji Restoration. The IJN continued to develop its strength and commissioned a number of warships from British and European shipyards, first ironclads and later armored cruisers. These ships engaged the Chinese Beiyang fleet which was superior on paper at least at the Battle of the Yalu River. Thanks to superior short-range firepower, the Japanese fleet came off better, sinking or severely damaging eight ships and receiving serious damage to only four. The naval war was concluded the next year at the Battle of Weihaiwei, where the strongest remaining Chinese ships were surrendered to the Japanese.[83] | END ID: 103

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 104 | TITLE: Vermont | CONTENT: The following were either born in Vermont or resided there for a substantial period during their lives. | END ID: 104

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 105 | TITLE: Tunisia | CONTENT: Though it is relatively small in size, Tunisia has great environmental diversity due to its north-south extent. Its east-west extent is limited. Differences in Tunisia, like the rest of the Maghreb, are largely north-south environmental differences defined by sharply decreasing rainfall southward from any point. The Dorsal, the eastern extension of the Atlas Mountains, runs across Tunisia in a northeasterly direction from the Algerian border in the west to the Cape Bon peninsula in the east. North of the Dorsal is the Tell, a region characterized by low, rolling hills and plains, again an extension of mountains to the west in Algeria. In the Khroumerie, the northwestern corner of the Tunisian Tell, elevations reach 1,050 metres (3,440Â ft) and snow occurs in winter. | END ID: 105

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 106 | TITLE: Helmuth von Moltke the Elder | CONTENT: In 1889, Moltke made two audio recordings with Adelbert Theodor Wangemann, a German native who worked with Thomas Edison and had been sent to Europe with Edison's newly invented cylinder phonograph.[9] Moltke recorded excerpts from Shakespeare and Goethe on two cylinders, recordings which were lost until 1957 and were unidentified for decades after. On January 30, 2012, they were among a number of recordings revealed by the Thomas Edison National Historical Park. The two cylinders made by Moltke are the only known voice recordings of anyone born in the 18th century.[9] | END ID: 106

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 107 | TITLE: Mount Isa | CONTENT: The 2011 census found that 52.8% of residents were male and 47.2% were female.[39] However, a rumour has circulated that the ratio of males to females living in Mount Isa was five to one. Former Mayor John Molony drew international press attention in August 2008 when he told the Townsville Bulletin newspaper that Mount Isa's gender imbalance made it a good place for "not so attractive" women to live.[40][41] | END ID: 107

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 108 | TITLE: Windows Live Mesh | CONTENT: Microsoft announced on February 20, 2012 that Windows Live Mesh is set to be superseded by a new SkyDrive desktop application, where the cloud storage portion for the application will utilize the full 7 GB SkyDrive storage (or more if the user has purchased additional storage), rather than the limited 5 GB "SkyDrive synced storage" in the current version of Windows Live Mesh. However, the new SkyDrive desktop application will not support direct PC-to-PC synchronization, and must utilize the SkyDrive cloud storage for synchronization between two or more devices.[17][18] On August 7, 2012, Microsoft released Windows Essentials 2012, where it was announced that Windows Live Mesh would be removed and replaced by the SkyDrive for Windows desktop application if a user upgrades from Windows Live Essentials 2011.[19] | END ID: 108

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 109 | TITLE: Saint Basil's Cathedral | CONTENT: Contemporary commentators clearly identified the new building as Trinity Church, after its easternmost sanctuary;[15] the status of "katholikon" ("sobor", large assembly church) has not been bestowed on it yet: | END ID: 109

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 110 | TITLE: A Glimpse Inside the Mind of Charles Swan III | CONTENT: A Glimpse Inside the Mind of Charles Swan III is a 2013 American comedy-drama film directed, written and produced by Roman Coppola. It stars Charlie Sheen, Jason Schwartzman, Bill Murray, Katheryn Winnick and Patricia Arquette. | END ID: 110

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 111 | TITLE: Permanent residence (United States) | CONTENT: Note: This list excludes countries that allow visa-free travel with valid U.S. visas (for example, Costa Rica,[64] Dominican Republic,[65] Mexico,[66] Panama)[67] Also note that the Green Card holder might already have visa-free access to many destinations by virtue of the nationality already held. | END ID: 111

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 112 | TITLE: Rick and Morty | CONTENT: Rick and Morty is an American adult animated science fiction comedy series created by Justin Roiland and Dan Harmon for Cartoon Network's late-night programming block Adult Swim. The series follows the misadventures of cynical mad scientist Rick Sanchez and his fretful, easily influenced grandson Morty Smith, who split their time between domestic life and interdimensional adventures. The series premiered on December 2, 2013, and the third season concluded on October 1, 2017. A fourth season has been mentioned, first by Harmon in a September 2017 interview, and later in the post-credits scene of the third season's finale. However, as of April 2018, its future remains uncertain. | END ID: 112

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 113 | TITLE: Jade Cole | CONTENT: Cole was born in Pittsburgh, Pennsylvania,[10] and she is of mixed ethnicity: her father is African-American, and her mother is Dutch.  Though her ambiguous ethnicity initially caused her annoyance when people questioned her about it, she learned to use it to her advantage in her modeling career.[11] | END ID: 113

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 114 | TITLE: The Vow (2012 film) | CONTENT: The Vow is based on the actual relationship of Kim and Krickitt Carpenter, who wrote a book about their marriage, also known as The Vow. Ten weeks after their wedding on 18 September 1993, the couple was involved in a serious car accident. Krickitt suffered a brain trauma, which erased all memories of her romance with Kim as well as their marriage. Kim was still deeply in love with his wife, although she viewed him as a stranger after the accident.[2] | END ID: 114

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 115 | TITLE: Personality disorder | CONTENT: The most recent fifth edition of the Diagnostic and Statistical Manual of Mental Disorders stresses a personality disorder is an enduring and inflexible pattern of long duration leading to significant distress or impairment and is not due to use of substances or another medical condition. The DSM-5 lists personality disorders in the same way as other mental disorders, rather than on a separate 'axis', as previously.[16] | END ID: 115

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 116 | TITLE: Aloha | CONTENT: From Chapter 5 of HawaiÊ»i Revised Statutes: | END ID: 116

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 117 | TITLE: Arsenal Stadium | CONTENT: Arsenal's clock was moved from Highbury to the outer side of the new stadium, with a new larger version of the feature added inside the ground in August 2010. At the same time as the unveiling of the new clock, the south stands at the venue were also renamed Clock End inline with the same name previously used at Highbury.[42][43] | END ID: 117

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 118 | TITLE: Salary cap | CONTENT: The cap was first introduced for the 1994 season and was initially $34.6 million. Both the cap and the floor are adjusted annually based on the league's revenues, and they have increased each year. In 2009, the final capped year under that agreement, the cap was $128 million per team, while the floor was 87.6% of the cap. Using the formula provided in the league's collective bargaining agreement, the floor in 2009 was $112.1 million. Under the NFL's agreement with the NFLPA, the effects on the salary cap of guaranteed payments (such as signing bonuses) are, with a few rare exceptions, prorated evenly over the term of the contract. | END ID: 118

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 119 | TITLE: Fox Broadcasting Company | CONTENT: Fox launched in Croatia on October 15, 2012. Operated by Fox International Channels Bulgaria, all of Fox's channels (Fox, Fox Life, Fox Crime, Fox Movies, 24Kitchen, NatGeo (both SD and HD), NatGeo Wild (also HD and SD) and BabyTV) carry programming identical to that available on its Serbian channels. Most of them, with the exception of Nat Geo HD and BabyTV, feature subtitled promos and program content. All of the channels, except for BabyTV, are broadcast in 16:9 widescreen, while Fox has plans to offer an HD feed. | END ID: 119

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 120 | TITLE: Killer whale | CONTENT: Information for offshore regions and warmer waters is more scarce, but widespread, if not frequent, sightings indicate the killer whale can survive in most water temperatures. They have been sighted, for example, in the Mediterranean, the Arabian Sea, the Gulf of Mexico and the Indian Ocean around the Seychelles.[90] In the Mediterranean, killer whales are considered "visitors" with the exception of one small population which lives in the Strait of Gibraltar.[104] Except for northeastern basins from Black Sea, records have been among almost entire basin including Aegean[105] and Levantine basins such as off Israel.[106] A distinct population may also exist in Papua New Guinea.[107][108] Distributions and abundances in other Asian waters are very unclear, only with sightings time to time have been reported, such as off Phuket[109] and Mergui Archipelago.[110] | END ID: 120

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 121 | TITLE: Karim Benzema | CONTENT: On 6 December 2006, he told RMC about his possible selection for the Algerian team: "It's my parents' country, it's in my heart. But good after sporting, it's true that I will play in French team. I will be always present for the French team. Then it's more for the sporting side, because Algeria is my country, here, my parents come from there. After, France ... It's more sporty, that's it."[302] Benzema drew some criticism for these comments, as well as for his reluctance to sing the French national anthem, "La Marseillaise", before each match with the national team.[303][304] | END ID: 121

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 122 | TITLE: Skat (card game) | CONTENT: The player could have bid up to that value (110) during the auction. In practice this would have been too risky because only the â™¦J in the Skat increased the length of Matadors jack strait to 7. | END ID: 122

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 123 | TITLE: Indira Gandhi | CONTENT: Indira formed her government with Morarji Desai as deputy prime minister and finance minister. At the beginning of her first term as prime minister, Indira was widely criticized by the media and the opposition as a "Goongi goodiya" (Hindi word for a dumb doll or puppet) of the Congress party bosses who had got her elected and tried to constrain her.[29][30] | END ID: 123

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 124 | TITLE: When You Say Nothing at All | CONTENT: RCA released "When You Say Nothing at All" as the follow-up single to the title song of Whitley's Don't Close Your Eyes album. The former song already had hit No. 1 on the Billboard Hot Country Singles chart, his first chart-topper after three prior singles made the top 10.[3] "When You Say Nothing at All" entered the Hot Country Singles chart on September 17, 1988, at No. 61, and gradually rose to the top, where it stayed for two weeks at the end of the year.[1][2] It was the second of five consecutive chart-topping singles for Whitley, who did not live to see the last two, as he died on May 9, 1989 of alcohol poisoning.[3] "Keith did a great job singin' that song," co-composer Schlitz told author Tom Roland. "He truly sang it from the heart."[2] In 2004, Whitley's original was ranked 12th among CMT's 100 Greatest Love Songs.[4] It was sung by Sara Evans on the show. As of February 2015, the song has sold 599,000 digital copies in the US after it became available for download.[5] | END ID: 124

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 125 | TITLE: Jack Lord | CONTENT: John Joseph Patrick Ryan (December 30, 1920 – January 21, 1998), best known by his stage name, Jack Lord, was an American television, film and Broadway actor and director and producer. He was known for his starring role as Steve McGarrett in the CBS television program Hawaii Five-O, which ran from 1968 to 1980. | END ID: 125

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 126 | TITLE: Turkish invasion of Cyprus | CONTENT: On 23 July 1974 the Greek military junta collapsed mainly because of the events in Cyprus. Greek political leaders in exile started returning to the country. On 24 July 1974 Constantine Karamanlis returned from Paris and was sworn in as Prime Minister. He kept Greece from entering the war, an act that was highly criticized as an act of treason. Shortly after this Nikos Sampson renounced the presidency and Glafcos Clerides temporarily took the role of president.[95] | END ID: 126

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 127 | TITLE: Multilayer perceptron | CONTENT: Using gradient descent, the change in each weight is | END ID: 127

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 128 | TITLE: Geology of the Moon | CONTENT: The lunar maria represent ancient flood basaltic eruptions. In comparison to terrestrial lavas, these contain higher iron abundances, have low viscosities, and some contain highly elevated abundances of the titanium-rich mineral ilmenite. The majority of basaltic eruptions occurred between about 3 and 3.5 Ga ago, though some mare samples have ages as old as 4.2 Ga, and the youngest (based on the method of crater counting) are believed to have erupted only 1 billion years ago. Along with mare volcanism came pyroclastic eruptions, which launched molten basaltic materials hundreds of kilometres away from the volcano. A large portion of the mare formed, or flowed into, the low elevations associated with the nearside impact basins. However, Oceanus Procellarum does not correspond to any known impact structure, and the lowest elevations of the Moon within the farside South Pole-Aitken basin are only modestly covered by mare (see lunar mare for a more detailed discussion). | END ID: 128

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 129 | TITLE: Benazir Bhutto | CONTENT: When Bhutto was five, her father became the cabinet minister for energy, and when she was nine he became the country's foreign minister.[23] From an early age, she was exposed to foreign diplomats and figures who were visiting her father, among them Zhou Enlai, Henry Kissinger, and Hubert Humphrey.[24] When she was thirteen, he resigned from the government and a year later established his own political party, the Pakistan People's Party (PPP).[25] The PPP used the motto "Islam is our faith, democracy is our policy, socialism is our economy. All power to the people."[26] It employed a populist strategy to attract votes, promising "roti, kapra aur makan" (bread, clothes, and housing) for every Pakistani and insisting that the disputed territory of Kashmir would be transferred from Indian to Pakistani control.[26] Benazir immediately joined.[23] Amid riots against the government of President Ayub Khan, in 1968 Zulfikar was arrested and imprisoned for three months, during which he wrote to Benazir to encourage her studies.[27] | END ID: 129

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 130 | TITLE: We Are the World | CONTENT: "We Are the World" is a song and charity single originally recorded by the supergroup United Support of Artists (USA) for Africa in 1985. It was written by Michael Jackson and Lionel Richie (with arrangements by Michael Omartian) and produced by Quincy Jones for the album We Are the World. With sales in excess of 20 million copies, it is one of the fewer than 30 all-time physical singles to have sold at least 10 million copies worldwide. | END ID: 130

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 131 | TITLE: SAT | CONTENT: On March 5, 2014, the College Board announced its plan to redesign the SAT in order to link the exam more closely to the work high school students encounter in the classroom.[7] The new exam was administered for the first time in March 2016.[81] Some of the major changes are: an emphasis on the use of evidence to support answers, a shift away from obscure vocabulary to words that students are more likely to encounter in college and career, a math section that is focused on fewer areas, a return to the 1600-point score scale, an optional essay, and the removal of penalty for wrong answers (rights-only scoring).[82] To combat the perceived advantage of costly test preparation courses, the College Board announced a new partnership with Khan Academy to offer free online practice problems and instructional videos.[7] | END ID: 131

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 132 | TITLE: Gravity | CONTENT: Gravity, or gravitation, is a natural phenomenon by which all things with mass are brought toward (or gravitate toward) one another, including objects ranging from atoms and photons, to planets and stars. Since energy and mass are equivalent, all forms of energy (including light) cause gravitation and are under the influence of it. On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. The gravitational attraction of the original gaseous matter present in the Universe caused it to begin coalescing, forming stars – and for the stars to group together into galaxies – so gravity is responsible for many of the large scale structures in the Universe. Gravity has an infinite range, although its effects become increasingly weaker on farther objects. | END ID: 132

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 133 | TITLE: Serial (literature) | CONTENT: Serialized fiction surged in popularity during Britain's Victorian era, due to a combination of the rise of literacy, technological advances in printing, and improved economics of distribution.[3]:34 Most Victorian novels first appeared as installments in monthly or weekly periodicals.[3]:13 The wild success of Charles Dickens's The Pickwick Papers, first published in 1836, is widely considered to have established the viability and appeal of the serialized format within periodical literature. During that era, the line between "quality" and "commercial" literature was not distinct.[3]:31 Other famous writers who wrote serial literature for popular magazines were Wilkie Collins, inventor of the detective novel withThe Moonstone and Sir Arthur Conan Doyle, who created the Sherlock Holmes stories originally for serialization in The Strand magazine. | END ID: 133

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 134 | TITLE: Orange (colour) | CONTENT: The orange colour of carrots, pumpkins, sweet potatoes, oranges, and many other fruits and vegetables comes from carotenes, a type of photosynthetic pigment. These pigments convert the light energy that the plants absorb from the sun into chemical energy for the plants' growth. The carotenes themselves take their name from the carrot.[22] Autumn leaves also get their orange colour from carotenes. When the weather turns cold and production of green chlorophyll stops, the orange colour remains. | END ID: 134

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 135 | TITLE: Puerto Rico | CONTENT: Originally populated by the indigenous Taíno people, the island was claimed in 1493 by Christopher Columbus for Spain during his second voyage. Later it endured invasion attempts from the French, Dutch, and British. Four centuries of Spanish colonial government influenced the island's cultural landscapes with waves of African slaves, Canarian, and Andalusian settlers. In the Spanish Empire, Puerto Rico played a secondary, but strategic role when compared to wealthier colonies like Peru and the mainland parts of New Spain.[22][23] Spain's distant administrative control continued up to the end of the 19th century, helping to produce a distinctive creole Hispanic culture and language that combined elements from the Native Americans, Africans, and Iberians.[24] In 1898, following the Spanish–American War, the United States acquired Puerto Rico under the terms of the Treaty of Paris. The treaty took effect on April 11, 1899.[4] | END ID: 135

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 136 | TITLE: Business continuity | CONTENT: Planning, prevention, and preparation are a key part of any business continuity management system and have direct read across from civil contingencies planning. The activity begins with understanding the business to identify potential risks and threats to critical business activities both internally and from the external environment. It is also advisable to examine the resilience of suppliers. | END ID: 136

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 137 | TITLE: Just war theory | CONTENT: A 2017 study found that the just war tradition can be traced as far back as to Ancient Egypt, "demonstrating that just war thought developed beyond the boundaries of Europe and existed many centuries earlier than the advent of Christianity or even the emergence of Greco-Roman doctrine."[8] | END ID: 137

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 138 | TITLE: Classification of percussion instruments | CONTENT: By far the most common way of classifying percussion is by the style or tradition with which it is most closely associated. | END ID: 138

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 139 | TITLE: Experimental film | CONTENT: The film society and self-financing model continued over the next two decades, but by the early 1960s, a different outlook became perceptible in the work of American avant-garde filmmakers. Artist Bruce Conner created early examples such as A Movie (1958) and Cosmic Ray (1962). As P. Adams Sitney has pointed out, in the work of Stan Brakhage and other American experimentalists of early period, film is used to express the individual consciousness of the maker, a cinematic equivalent of the first person in literature. Brakhage's Dog Star Man (1961â€“64) exemplified a shift from personal confessional to abstraction, and also evidenced a rejection of American mass culture of the time. On the other hand, Kenneth Anger added a rock sound track to his Scorpio Rising (1963) in what is sometimes said to be an anticipation of music videos, and included some camp commentary on Hollywood mythology. Jack Smith and Andy Warhol incorporated camp elements into their work, and Sitney posited Warhol's connection to structural film. | END ID: 139

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 140 | TITLE: Column (database) | CONTENT: Some examples of popular databases include: | END ID: 140

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 141 | TITLE: List of The Amazing World of Gumball characters | CONTENT: Harold is the father of Tobias Wilson. He seems to have been making fun of Richard Watterson ever since he played a prank on him in high school, as seen in "The Cycle." | END ID: 141

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 142 | TITLE: Rotator cuff tear | CONTENT: The cuff is responsible for stabilizing the glenohumeral joint, abducting, externally rotating, and internally rotating the humerus. When shoulder trauma occurs, these functions can be compromised. Because individuals are highly dependent on the shoulder for many activities, overuse of the muscles can lead to tears, the vast majority again occurring in the supraspinatus tendon. | END ID: 142

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 143 | TITLE: River | CONTENT: The Strahler Stream Order ranks rivers based on the connectivity and hierarchy of contributing tributaries. Headwaters are first order while the Amazon River is twelfth order. Approximately 80% of the rivers and streams in the world are of the first and second order. | END ID: 143

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 144 | TITLE: For the Love of Strange Medicine | CONTENT: Journey released their ninth studio album Raised on Radio in 1986, which was Steve Perry' sixth album as lead singer. The band subsequently went on a hiatus in 1987. After the split Perry "didn't feel the passion" for writing and recording music, but eventually began writing songs for the album with musicians Lincoln Brewster, Paul Taylor, and Moyes Lucas.[1] | END ID: 144

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 145 | TITLE: Hurricane Gloria | CONTENT: At the same time Gloria was making landfall on Long Island, a storm warning was issued for western New Brunswick and Nova Scotia.[6][10] Across Atlantic Canada, the threat of Hurricane Gloria caused many citizens to rely on American media for storm coverage.[16] | END ID: 145

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 146 | TITLE: Winston Churchill | CONTENT: In June 1944, the Allied Forces invaded Normandy and pushed the Nazi forces back into Germany on a broad front over the coming year. After being attacked on three fronts by the Allies, and in spite of Allied failures, such as Operation Market Garden, and German counter-attacks, including the Battle of the Bulge, Germany was eventually defeated. On 7 May 1945 at the SHAEF headquarters in Rheims the Allies accepted Germany's surrender. On the same day in a BBC news flash John Snagge announced that 8 May would be Victory in Europe Day.[425] On Victory in Europe Day, Churchill broadcast to the nation that Germany had surrendered and that a final ceasefire on all fronts in Europe would come into effect at one minute past midnight that night.[426][427] | END ID: 146

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 147 | TITLE: Holy Week | CONTENT: Holy Week begins with Palm Sunday, which may also be known as Passion Sunday in some denominations. Traditionally, Palm Sunday commemorates the Triumphal entry into Jerusalem described in all four canonical gospels. As described in the accounts, Jesus's entry into Jerusalem was noted by the crowds present who shouted praises and waved palm branches. In the Roman Rite, before 1955 it was known simply as Palm Sunday, and the preceding Sunday as Passion Sunday. From 1955 to 1971 it was called Second Sunday in Passiontide or Palm Sunday. Among Lutherans and Anglicans, the day is known as the Sunday of the Passion: Palm Sunday.[6] | END ID: 147

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 148 | TITLE: Billy Bob Thornton | CONTENT: In 2000, Thornton appeared in Travis Tritt's music video for the song "Modern Day Bonnie and Clyde". His screen persona has been described by the press as that of a "tattooed, hirsute man's man".[15] He appeared in several major film roles following the success of Sling Blade, including 1998's Armageddon and A Simple Plan. In 2001, he directed Daddy and Them while securing starring roles in three Hollywood films: Monster's Ball, Bandits, and The Man Who Wasn't There, for which he received many awards.[citation needed] | END ID: 148

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 149 | TITLE: Sicily | CONTENT: Sicily has four universities: | END ID: 149

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 150 | TITLE: Educational technology | CONTENT: Educational technology is the use of both physical hardware and educational theoretics. It encompasses several domains, including learning theory, computer-based training, online learning, and, where mobile technologies are used, m-learning.[2] Accordingly, there are several discrete aspects to describing the intellectual and technical development of educational technology: | END ID: 150

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 151 | TITLE: 2014–15 Chelsea F.C. season | CONTENT: A dominating performance against West Bromwich Albion sent kept Chelsea seven points clear at the top of the table, thanks to goals from Diego Costa and Eden Hazard.[61] The Blues secured qualification for the Second Round of the Champions League after hammering Schalke 04 0–5 at the Veltins-Arena.[62] Chelsea's final game in November was a 0–0 stalemate away at Sunderland. Despite Chelsea failing to score for the first time in the season, they went six points clear of defending champions Manchester City at November's end.[63] | END ID: 151

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 152 | TITLE: Rockford Central High School | CONTENT: The Annual, or yearbook as it is called now, was entitled "The Owl".[2] It was founded in 1890 and has been published continually since 1892. Rockford was the second high school in the country to establish a yearbook. | END ID: 152

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 153 | TITLE: Macromolecule | CONTENT: Because of their size, macromolecules are not conveniently described in terms of stoichiometry alone. The structure of simple macromolecules, such as homopolymers, may be described in terms of the individual monomer subunit and total molecular mass. Complicated biomacromolecules, on the other hand, require multi-faceted structural description such as the hierarchy of structures used to describe proteins. In British English, the word "macromolecule" tends to be called "high polymer". | END ID: 153

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 154 | TITLE: Elastic collision | CONTENT: The collisions    of atoms are elastic collisions (Rutherford backscattering is one example). | END ID: 154

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 155 | TITLE: List of Care Bear characters | CONTENT: He also teaches the value of good sportsmanship. | END ID: 155

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 156 | TITLE: Abul Kalam Azad | CONTENT: Maulana Azad is considered one of the greatest Urdu writers of the 20th century. He has written many books including India Wins Freedom, Ghubar-e-Khatir, Tazkirah, Tarjumanul Quran, etc. | END ID: 156

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 157 | TITLE: South Sydney Rabbitohs | CONTENT: The club was formed in 1908 as one of the founding members of the New South Wales Rugby Football League, making them one of Australia's oldest rugby league teams. The Rabbitohs were formed, under their original 1908 articles of association with the NSWRL competition, to represent the Sydney municipalities of Redfern, Alexandria, Zetland, Waterloo, Mascot and Botany. They are one of only two foundation clubs still present in the NRL, the other being the Sydney Roosters.[4] The South Sydney District Rugby League Football Club is currently a subsidiary company 75% owned by Blackcourt League Investments which is, in turn, 50% owned by the actor Russell Crowe and 50% owned by James Packer's Consolidated Press Holdings; the other 25% is owned by the financial Members of the club.[5] | END ID: 157

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 158 | TITLE: Lykan HyperSport | CONTENT: The Lykan Hypersport is a Lebanese limited production supercar built by W Motors, a United Arab Emirates based company, founded in 2012 in Lebanon with the collaboration of Lebanese, French[1] and Italian engineers.[2] It is the first supercar to be produced in the Middle East, and is featured in the film Furious 7, and the video games Project CARS, Driveclub, Asphalt 8: Airborne, Asphalt Nitro, Forza Motorsport 6, and GT Racing 2: The Real Car Experience, Forza Horizon 3, CSR Racing and CSR Racing 2.[3] The Lykan can also be briefly seen in the second Fate of the Furious trailer, however, the Lykan does not make an appearance, the footage is from the seventh movie, Fast and Furious 7. It is the first car to be designed and produced indigenously in the Arab World.[4] | END ID: 158

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 159 | TITLE: Livy | CONTENT: Titus Livius died in his home city of Patavium in either (see below) AD 12 or 17; the latter would have been three years after the death of the emperor Augustus.[3] | END ID: 159

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 160 | TITLE: Gas giant | CONTENT: A gas dwarf could be defined as a planet with a rocky core that has accumulated a thick envelope of hydrogen, helium and other volatiles, having as result a total radius between 1.7 and 3.9 Earth-radii.[10][11] | END ID: 160

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 161 | TITLE: J. G. Hertzler | CONTENT: Hertzler began his acting career in the 1970s, doing mostly stage acting and appearing in some films. He guest starred in a few episodes for different television shows before landing the part of Alcalde Ignacio De Soto in the early 1990s show Zorro. In addition to Deep Space Nine, Hertzler has appeared on several other Star Trek shows, written two Star Trek novels, and has made appearances at Star Trek and science fiction conventions. Hertzler lives in the Finger Lakes region of New York where he was a lecturer at Cornell's theater department, and has been active in the area's regional politics, as well as writing a screenplay. | END ID: 161

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 162 | TITLE: Coach Trip | CONTENT: Day 22 of Series 14 entitled "Road to Ibiza" saw the voting rules again being amended after the announcement about the double yellow terror whichever two couples receives the most votes will both receive a yellow card which is somewhat similar to when two couples received a yellow card from tied votes from not only earlier series of Coach Trip but also from day 12 of the same series Day 5 of Series 15 entitled "Road to Marbs" had the same voting rules as Day 22 of "Road to Ibiza" as 2 couples each received 2 yellow cards each from tied votes. | END ID: 162

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 163 | TITLE: Living in the Material World | CONTENT: With Living in the Material World, Harrison achieved the Billboard double for a second time when "Give Me Love" hit the top position during the album's stay at number 1[74] â€“ the only one of his former bandmates to have done it even once being McCartney, with the recent "My Love" and Red Rose Speedway.[145][148] Harrison carried out no supporting promotion for Material World; "pre-recorded tapes" were issued to BBC Radio 1 and played repeatedly on the show Radio One Club, but his only public appearance in Britain was to accompany Prabhupada on a religious procession through central London, on 8 July.[149] According to author Bill Harry, the album sold over 3 million copies worldwide.[150] | END ID: 163

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 164 | TITLE: Influenza vaccine | CONTENT: As the death rate is also high among infants who catch influenza, the household contacts and caregivers of infants should be vaccinated to reduce the risk of passing an influenza infection to the infant.[28] | END ID: 164

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 165 | TITLE: Montana | CONTENT: Montana is one of the nine Mountain States, located in the north of the region known as the Western United States. It borders North Dakota and South Dakota to the east. Wyoming is to the south, Idaho is to the west and southwest, [15] and three Canadian provinces, British Columbia, Alberta, and Saskatchewan, are to the north. | END ID: 165

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 166 | TITLE: Chagos Archipelago sovereignty dispute | CONTENT: 1. There is no objection to Ministers referring to points contained in paragraph 22 of enclosure to Secret despatch No. 423 of 6 October so long as qualifications contained in paragraphs 5 and 6 of the despatch are borne in mind. | END ID: 166

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 167 | TITLE: Northern Pacific Railway | CONTENT: The North Coast Limited was a famous passenger train operated by the Northern Pacific Railway between Chicago and Seattle via Butte, Montana and Homestake Pass. It commenced service on April 29, 1900, served briefly as a Burlington Northern train after the merger on March 2, 1970, and ceased operation the day before Amtrak began service (April 30, 1971). The Chicago Union Station to Saint Paul leg of the train's route was operated by the Chicago, Burlington and Quincy Railroad along its Mississippi River mainline through Wisconsin. The North Coast Limited was the Northern Pacific's flagship train and the Northern Pacific itself was built along the trail first blazed by Lewis and Clark. | END ID: 167

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 168 | TITLE: Clock management | CONTENT: Clock management is also a component of the game of basketball. In that sport, the rules governing the game clock are simpler; the clock stops when the ball is dead and runs when it is live. Most clock management in basketball centers around both the game clock and the shot clock. An offense nearing the end of a game and holding a slim lead will attempt to use up as much of both clocks as possible before shooting the ball to give the opposing team as little time as possible to respond. | END ID: 168

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 169 | TITLE: Cold War (1953–1962) | CONTENT: In the meantime, however, attention was being diverted elsewhere in Asia. The continuing pressure from the "China lobby" or "Asia firsters," who had insisted on active efforts to restore Chiang Kai-shek to power was still a strong domestic influence on foreign policy. In April 1953 Senator Robert A. Taft and other powerful Congressional Republicans suddenly called for the immediate replacement of the top chiefs of the Pentagon, particularly the Chairman of the Joint Chiefs of Staff, Omar Bradley. To the so-called "China lobby" and Taft, Bradley was seen as having leanings toward a Europe-first orientation, meaning that he would be a possible barrier to new departures in military policy that they favored. Another factor was the vitriolic accusations of McCarthyism, where large portions of the U.S. government allegedly contained covert communist agents or sympathizers. But after the mid-term elections in 1954–and censure by the Senate–the influence of Joseph McCarthy ebbed after his unpopular accusations against the Army. | END ID: 169

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 170 | TITLE: Albrecht Dürer | CONTENT: His famous series of sixteen great designs for the Apocalypse[11] is dated 1498, as is his engraving of St. Michael Fighting the Dragon. He made the first seven scenes of the Great Passion in the same year, and a little later, a series of eleven on the Holy Family and saints. The Seven Sorrows Polyptych, commissioned by Frederick III of Saxony in 1496, was executed by Dürer and his assistants c. 1500. Around 1503–1505 he produced the first seventeen of a set illustrating the Life of the Virgin, which he did not finish for some years. Neither these, nor the Great Passion, were published as sets until several years later, but prints were sold individually in considerable numbers.[6] | END ID: 170

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 171 | TITLE: Culture of Iran | CONTENT: The Persian year begins in the vernal equinox: if the astronomical vernal equinox comes before noon, then the present day is the first day of the Persian year. If the equinox falls after noon, then the next day is the official first day of the Persian year. The Persian Calendar, which is the official calendar of Iran, is a solar calendar with a starting point that is the same as the Islamic calendar. According to the Iran Labor Code, Friday is the weekly day of rest. Government official working hours are from Saturday to Wednesday (from 8 am to 4 pm).[5] | END ID: 171

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 172 | TITLE: LisaRaye McCoy | CONTENT: McCoy has a daughter named Kai Morae Pace (b. 1989) from a previous relationship with Kenji Pace.  In 1992, McCoy married Tony Martin. They divorced in 1994.  In April 2006, McCoy married Michael Misick, who had been elected the Premier of the Turks and Caicos Islands (a position previously known as "Chief Minister of the Turks and Caicos Islands") in 2003.  They were married in a lavish ceremony before 300 guests, followed by a three-week honeymoon to Jerusalem, Bali and Dubai.[23] During their marriage, McCoy's title was "First Lady of Turks and Caicos." In August 2008, Premier Misick released a statement announcing that he and McCoy were divorcing.[24] Misick resigned from office in March 2009 after an investigation found "clear signs of corruption" involving selling off public land to fund his own investments. He fled Turks and Caicos, and was eventually arrested in Brazil and extradited back to the islands to stand trial.[25] | END ID: 172

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 173 | TITLE: University of Maryland, College Park | CONTENT: Men's basketball is one of the most popular sports at the university.[184] Long-time head coach Lefty Driesell began the now nationwide tradition of "Midnight Madness" in 1971.[185] Beginning in 1989, alumnus Gary Williams revived the program, which was struggling in the wake of Len Bias's death and NCAA rules infractions. Williams led Maryland basketball to national prominence with two Final Four appearances, and in 2002, a national championship. On February 7, 2006, Gary Williams won his 349th game to surpass Driesell and became Maryland's all-time leader among basketball coaches. In May 2011, Williams retired as head coach, which allowed for the entrance of the new head coach, Mark Turgeon. The court at XFINITY Center was named in honor of the beloved coach, Gary Williams. Maryland football is also popular at the university.[184] The Terrapins were awarded the national championship by the wire services in 1953, and in 1951, by several retroactive selectors. | END ID: 173

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 174 | TITLE: Primary and secondary legislation | CONTENT: Forms of secondary legislation in the United Kingdom include:[4] | END ID: 174

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 175 | TITLE: Elizabeth Patterson (actress) | CONTENT: Mary Elizabeth Patterson (November 22, 1874 â€“ January 31, 1966) was an American theatre, film, and television character actress who gained popular recognition late in her career playing the elderly neighbor Matilda Trumbull on the television comedy series I Love Lucy.[1] | END ID: 175

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 176 | TITLE: List of FIFA World Cup records | CONTENT: Biggest margin of victory | END ID: 176

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 177 | TITLE: Health care finance in the United States | CONTENT: In 2009, the United States federal, state and local governments, corporations and individuals, together spent $2.5 trillion, $8,047 per person, on health care.[17] This amount represented 17.3% of the GDP, up from 16.2% in 2008.[17] Health insurance costs are rising faster than wages or inflation,[18] and medical causes were cited by about half of bankruptcy filers in the United States in 2001.[19] | END ID: 177

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 178 | TITLE: A Dictionary of the English Language | CONTENT: Several more dictionaries followed: in Latin, English, French and Italian. Benjamin Martin's Lingua Britannica Reformata (1749) and Ainsworth's Thesaurus Linguae Latinae (1737) are both significant, in that they define entries in separate senses, or aspects, of the word. In English (among others), John Cowell's Interpreter, a law dictionary, was published in 1607, Edward Phillips' The new world of English words came out in 1658 and a dictionary of 40,000 words had been prepared in 1721 by Nathan Bailey, though none was as comprehensive in breadth or style as Johnson's. | END ID: 178

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 179 | TITLE: So You Think You Can Dance (U.S. season 2) | CONTENT: One hundred and sixteen dancers were invited for a week of training to the Aladdin hotel (now Planet Hollywood Resort and Casino) in Las Vegas, Nevada. This training included hip-hop choreography taught by Shane Sparks, samba choreography taught by Mary Murphy with assistance from season 1 contestant Artem Chigvinsev, a contemporary routine taught by Mia Michaels, and training from Brian Friedman, who described his choreography as a fusion of jazz and hip-hop.[1] The original group of dancers was eventually whittled down to 41, from which the judges chose their top 20. | END ID: 179

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 180 | TITLE: Questioned document examination | CONTENT: The Forensic Science Society (UK) provides their members with the opportunity to obtain a Professional Postgraduate Diploma in forensic disciplines, including Questioned Document Examination.[7] The program is accredited by the University of Strathclyde. Successful applicants are entitled to use the postnominal 'FSSocDip'. | END ID: 180

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 181 | TITLE: Leung Chun-ying | CONTENT: Despite the centuries-long history of Cantonese as the de facto spoken language of Hong Kong, Leung made his inaugural speech in Mandarin, spoken in Mainland China. This was in stark contrast to his predecessor, Sir Donald Tsang, who made his inaugural speech in Cantonese in July 2007.[49] | END ID: 181

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 182 | TITLE: Optic nerve | CONTENT: Cerebral peduncle, optic chasm, cerebral aqueduct. Inferior view. Deep dissection. | END ID: 182

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 183 | TITLE: Gerald R. Ford-class aircraft carrier | CONTENT: The receiver has high dynamic range to support high clutter levels caused by close returns from range-ambiguous Doppler effect waveforms. The receiver has both narrow band and wideband channels, as well as multichannel capabilities to support monopulse radar processing and side lobe blanking. The receiver generates digital data and sends the data to the signal processors. | END ID: 183

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 184 | TITLE: Chinese Immigration Act, 1923 | CONTENT: Before 1923, Chinese immigration was heavily controlled by the Chinese Immigration Act of 1885, which imposed a hefty head tax on all immigrants from China. After various members of the federal and some provincial governments (especially British Columbia) put pressure on the federal government to discourage Chinese immigration, the Chinese Immigration Act was passed. It went into effect on July 1, 1923. The act banned Chinese immigrants from entering Canada except those under the following titles: | END ID: 184

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 185 | TITLE: James Webb Space Telescope | CONTENT: James Webb Space Telescope insignia | END ID: 185

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 186 | TITLE: XMLHttpRequest | CONTENT: Upon successful initialization of a request, the setRequestHeader method of the XMLHttpRequest object can be invoked to send HTTP headers with the request. | END ID: 186

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 187 | TITLE: Tsurani | CONTENT: Only one of every five persons inducted into training to become a Great One makes it to this ultimate goal and takes his place in the Assembly of Magicians, while those who fail die in the process. Women are not permitted to become Great Ones, female children that display any magical ability are abruptly removed from their homes by the Assembly and subsequently murdered, unbeknown to the rest of the Empire. | END ID: 187

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 188 | TITLE: North Carolina Tar Heels men's basketball | CONTENT: Guthridge retired in 2000 and North Carolina turned to Matt Doherty, the head coach at Notre Dame and a player on the 1982 championship team, to lead the Tar Heels.[31] Doherty had little success while at North Carolina. In his first season, the Heels were ranked #1 in the polls in the middle of the Atlantic Coast Conference schedule and finished with a 26–7 record. But Doherty's second season was the worst in recent history as the Tar Heels finished the season with a record of 8–20, missing postseason play entirely for the first time since the 1965–66 season (including a record 27 straight NCAA Tournament appearances) and finishing with a losing record for the first time since 1962 (Dean Smith's first year as coach). They also finished 4–12 in the ACC—only the program's second losing ACC record ever. The 12 losses were six more than the Tar Heels had ever suffered in a single season of ACC play, and placed them in a tie for 7th place—the program's first finish below fourth place ever. The season also saw the end of UNC's run of 31 straight 20-win seasons and 35 straight seasons of finishing third or higher in the ACC. After bringing in one of the top 5 incoming classes for the 2002–2003 season, the Tar Heels started the season by knocking off a top 5 Kansas team and going on to win the Preseason NIT and returning to the AP top 25. North Carolina went on to finish the season 17–15, missing the NCAA tournament. Matt Doherty led the Tar Heels to the third round of the NIT, where they ended their season with a loss to Georgetown. | END ID: 188

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 189 | TITLE: Buffalo Bills | CONTENT: The Bills began competitive play in 1960 as a charter member of the American Football League led by head coach Buster Ramsey and joined the NFL as part of the AFL–NFL merger in 1970.[9] The Bills won two consecutive American Football League titles in 1964 and 1965, but the club has yet to win a league championship since. | END ID: 189

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 190 | TITLE: Voyages of Christopher Columbus | CONTENT: A conjectural replica of the NiÃ±a | END ID: 190

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 191 | TITLE: Miami Dolphins | CONTENT: ("The Dolphin") On Friday, April 18, 1997, the first "official" mascot of the Miami Dolphins was introduced. The 7-foot mascot made his public debut on April 19 at Pro Player Stadium during the team's draft day party. The team then made a "Name the Mascot" contest that drew over 13,000 entries covering all 50 states and 22 countries. 529 names were suggested. The winning entry was announced at the annual Dolphins Awards Banquet on June 4, 1997. | END ID: 191

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 192 | TITLE: Culture during the Cold War | CONTENT: During the Cold War, films functioned as a means to influence and control public opinion internally.  The United States and the Soviet Union invested heavily in propaganda designed to influence the hearts and minds of people around the world, especially using motion pictures.[9]  Cold War films produced by both sides attempted to address different facets of the superpower conflict and sought to influence both domestic and foreign opinion. The gap between American and Soviet film gave the Americans a distinct advantage over the Soviet Union; America was readily prepared to utilize their cinematic achievements as a way to effectively impact the public opinion in a way the Soviet Union could not. Cinema, Americans hoped, would help close the gap caused by Soviet development of nuclear weapons and advancements in space technology.[10] The use of film as an effective form of widespread propaganda transformed cinema into another Cold War battlefront. | END ID: 192

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 193 | TITLE: Al Ahly SC | CONTENT: The Egyptian League championship began in 1948â€“49. Al Ahly won the inaugural competition, the first of nine successive national championship titles.[2] Following the deposing of King Farouk in the revolution of 1952, Ahly appointed Gamal Abdel Nasser as club president.[3] | END ID: 193

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 194 | TITLE: Flying buttress | CONTENT: As a lateral-support system, the flying buttress was developed during late antiquity and later flourished during the Gothic period (12th–16th c.) of architecture. Ancient examples of the flying buttress can be found on the Basilica of San Vitale in Ravenna and on the Rotunda of Galerius in Thessaloniki. The architectural-element precursors of the medieval flying buttress derive from Byzantine architecture and Romanesque architecture, in the design of churches, such as Durham Cathedral, where arches transmit the lateral thrust of the stone vault over the aisles; the arches were hidden under the gallery roof, and transmitted the lateral forces to the massive, outer walls. By the decade of 1160, architects in the Île-de-France region employed similar lateral-support systems that featured longer arches of finer design, which run from the outer surface of the clerestory wall, over the roof of the side aisles (hence are visible from the outside) to meet a heavy, vertical buttress rising above the top of the outer wall.[3] | END ID: 194

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 195 | TITLE: Lisbeth Salander | CONTENT: In The Girl With the Dragon Tattoo (2005), Lisbeth Salander is introduced as a gifted, but deeply troubled, researcher and computer hacker working for Milton Security. Her boss, Dragan Armansky, commissions her to research disgraced journalist Mikael Blomkvist at the behest of a wealthy businessman, Henrik Vanger. When Blomkvist finds out that Salander hacked his computer, he hires her to assist him in investigating the disappearance of Vanger's grandniece, Harriet, 40 years earlier. Salander uses her research skills to uncover a series of murders, dating back decades and tied to Harriet's disappearance. During the investigation, Salander and Blomkvist become lovers. | END ID: 195

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 196 | TITLE: Federal Art Project | CONTENT: Hundreds of thousands of artworks were commissioned under the Federal Art Project.[5] Many of the portable works have been lost, abandoned or given away as unauthorized gifts. As custodian of the work, which remains Federal property, the General Services Administration maintains an inventory[159] and works with the FBI and art community to identify and recover WPA art.[160] In 2010 it produced a 22-minute documentary about the WPA Art Recovery Project, "Returning Americaâ€™s Art to America", narrated by Charles Osgood.[161] | END ID: 196

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 197 | TITLE: Uluru | CONTENT: The Commonwealth Department of Environment's webpage advises:[16] | END ID: 197

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 198 | TITLE: Costa Rican cuisine | CONTENT: Chorreadas are not as common as many other traditional dishes. They are corn pancakes and are served for breakfast with sour cream.[4] | END ID: 198

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 199 | TITLE: Dewey Decimal Classification | CONTENT: Melvil Dewey (1851â€“1931) was an American librarian and self-declared reformer.[5] He was a founding member of the American Library Association and can be credited with the promotion of card systems in libraries and business.[6] He developed the ideas for his library classification system in 1873 while working at Amherst College library. He applied the classification to the books in that library, until in 1876 he had a first version of the classification. In 1876, he published the classification in pamphlet form with the title A Classification and Subject Index for Cataloguing and Arranging the Books and Pamphlets of a Library.[7] He used the pamphlet, published in more than one version during the year, to solicit comments from other librarians. It is not known who received copies or how many commented as only one copy with comments has survived, that of Ernest Cushing Richardson.[8] His classification system was mentioned in an article in the first issue of the Library Journal and in an article by Dewey in the Department of Education publication "Public Libraries in America" in 1876.[9] In March 1876, he applied for, and received copyright on the first edition of the index.[10] The edition was 44 pages in length, with 2,000 index entries, and was printed in 200 copies.[11] | END ID: 199

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 200 | TITLE: Rapid eye movement sleep | CONTENT: Erections of the penis (nocturnal penile tumescence or NPT) normally accompany REM sleep in rats and humans.[35] If a male has erectile dysfunction (ED) while awake, but has NPT episodes during REM, it would suggest that the ED is from a psychological rather than a physiological cause. In females, erection of the clitoris (nocturnal clitoral tumescence or NCT) causes enlargement, with accompanying vaginal blood flow and transudation (i.e. lubrication). During a normal night of sleep the penis and clitoris may be erect for a total time of from one hour to as long as three and a half hours during REM.[36] | END ID: 200

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 201 | TITLE: United Kingdom in the Eurovision Song Contest | CONTENT: If Scotland were to participate it is unknown whether or not England, Wales and Northern Ireland would show any interest in entering the Eurovision Song Contest independently as well, although S4C (the Welsh language media channel) has expressed an interest and, in addition, already holds a yearly national song contest called "Cân i Gymru" (Song for Wales).[43] S4C also considered a bid for the Junior Eurovision Song Contest 2008 but decided not to go ahead.[44] In 2009 MEP for Wales Jillian Evans stated her interest in securing Wales a place in the Eurovision Song Contest 2010, Wales could be represented by either BBC Cymru Wales, ITV Wales & West or S4C. There is a small campaign in Northern Ireland for a separate entrant and it could be represented by UTV or BBC Northern Ireland.[45] There are no plans currently for England to enter separately. | END ID: 201

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 202 | TITLE: Richard Keith (actor) | CONTENT: Keith Thibodeaux (born December 1, 1950) is a former American child actor of television and film and musician, best known for playing Little Ricky on the television sitcom's I Love Lucy and The Lucy-Desi Comedy Hour, his last name "Thibodeaux" which was Cajun French was changed by co-star Desi Arnaz, to "Keith" because his surname was more difficult to pronounce. He is the last living regular appearing cast member from I Love Lucy. | END ID: 202

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 203 | TITLE: Lieutenant governor (United States) | CONTENT: The positions are sometimes criticized for lacking duties and power and described by political insiders as "get up, read the paper, see if the governor is dead, if not, go back to sleep".[5] In the 2010 election for the Lieutenant Governor of Rhode Island, 40% of the vote was won by a perennial candidate who wanted to abolish the office,[6] saying "If you open up the dictionary to ‘sinecure,’ you have a picture of the lieutenant governor of Rhode Island". | END ID: 203

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 204 | TITLE: Assassin's Creed III: Liberation | CONTENT: Purchasing Assassin's Creed III for the PlayStation 3 gives the player the ability to connect Liberation and receive an exclusive mission to play in Liberation as Connor or Aveline, a Multiplayer Skin and an Ammunition Pouch. There was also a promotional DLC, titled Mysteries of the Bayou pack, that came with pre-orders of the game in PAL regions. It included an exclusive weapon, an alligator hunting hat, a Multiplayer Skin and Ammunition Pouches for smoke bombs and poison darts. | END ID: 204

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 205 | TITLE: Tom Clancy's Splinter Cell: Blacklist | CONTENT: Due to files recovered by Fisher in London (and made public by the president), the American public and Congress both believe that Iran is funding the Blacklist, and The United States and Iran are headed towards war. Looking for more solid proof of Iran's involvement, Fisher infiltrates Quds Force headquarters in Tehran(previously a U.S embassy). Inside, he discovers that Iran has nothing to do with the Blacklist, and that the files he recovered were planted by The Engineers in order to lure The United States into a war. Fisher narrowly escapes Tehran by using a Predator Drone to blow up the Iranian vehicles chasing him, causing an international incident. | END ID: 205

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 206 | TITLE: Life of Pi | CONTENT: Life of Pi is a Canadian fantasy adventure novel by Yann Martel published in 2001. The protagonist is Piscine Molitor "Pi" Patel, an Indian boy from Pondicherry who explores issues of spirituality and practicality from an early age. He survives 227 days after a shipwreck while stranded on a lifeboat in the Pacific Ocean with a Bengal tiger named Richard Parker. | END ID: 206

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 207 | TITLE: Wedding ring | CONTENT: In several traditions, the best man or maid of honour has the duty of keeping track of a couple's wedding rings and to produce them at the symbolic moment of the giving and receiving of the rings during the traditional marriage ceremony. In more elaborate weddings, a ring bearer (who is often part of the family of the bride or groom) may assist in the ceremonial parading of the rings into the ceremony, sometimes on a special cushion. | END ID: 207

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 208 | TITLE: Huntington's disease | CONTENT: The first likely description of the disease was in 1841 by Charles Oscar Waters.[7] The condition was described in further detail in 1872 by the physician George Huntington, after whom it is named.[7] The genetic basis was discovered in 1993 by an international collaborative effort led by the Hereditary Disease Foundation.[8][9] Research and support organizations began forming in the late 1960s to increase public awareness, to provide support for individuals and their families, and to promote research.[9][10] Current research directions include determining the exact mechanism of the disease, improving animal models to aid with research, testing of medications to treat symptoms or slow the progression of the disease, and studying procedures such as stem cell therapy with the goal of repairing damage caused by the disease.[8] | END ID: 208

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 209 | TITLE: Municipal Corporation of Delhi | CONTENT: The Municipal Corporation of Delhi (MCD) is a municipal corporation, an autonomous body that governs 8 of the 11 Districts of Delhi, in the state of Delhi, India. It was one of three municipalities in the National Capital Territory of Delhi, the others being New Delhi Municipal Council, and Delhi Cantonment Board. "The MCD was among the largest municipal bodies in the world providing civic services to more than estimated population of 11 million citizens in the capital city.[1] The municipal corporation covers an area of 1,397.3 km² (539.5 mi²). | END ID: 209

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 210 | TITLE: Central bank | CONTENT: In some countries, central banks may have other tools that work indirectly to limit lending practices and otherwise restrict or regulate capital markets. For example, a central bank may regulate margin lending, whereby individuals or companies may borrow against pledged securities. The margin requirement establishes a minimum ratio of the value of the securities to the amount borrowed. | END ID: 210

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 211 | TITLE: Business intelligence | CONTENT: Often[quantify] BI applications use data gathered from a data warehouse (DW) or from a data mart, and the concepts of BI and DW combine as "BI/DW"[5]
or as "BIDW". A data warehouse contains a copy of analytical data that facilitate decision support. | END ID: 211

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 212 | TITLE: Clef | CONTENT: In modern Gregorian chant notation, the C clef is written (on a four-line stave) in the form  and the F clef as | END ID: 212

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 213 | TITLE: First Indochina War | CONTENT: On November 14, 1951, the French seized Hòa Bình, 25 miles (40 km) west of the De Lattre Line, by a parachute drop and extended their perimeter. | END ID: 213

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 214 | TITLE: Luminosity | CONTENT: Imagine a point source of light of luminosity 

L{\displaystyle L}

that radiates equally in all directions. A hollow sphere centered on the point would have its entire interior surface illuminated. As the radius increases, the surface area will also increase, and the constant luminosity has more surface area to illuminate, leading to a decrease in observed brightness. | END ID: 214

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 215 | TITLE: The Book of Five Rings | CONTENT: "Flowing Water Cut" technique refers to if you come into a fight with an enemy of a similar level to you in swordsmanship. When attacking fast, Musashi notes that you will always be at stalemate, so like Stagnant water, you must cut as slowly as possible with your long sword. At the beginning of this technique you and your opponent will be searching for an opening within each other's defense. When your opponent either tries to push off your sword, or to hasten back as to disengage it, you must first expand your whole body and your mind. By moving your body first and then that of your sword, you will be able to strike powerfully and broadly with a movement that seems to reflect the natural flow of water. Ease and confidence will be attained when this technique is continuously practiced upon. | END ID: 215

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 216 | TITLE: List of A Certain Magical Index episodes | CONTENT: The first Blu-ray and DVD compilation for the series was released on January 23, 2009, with one scheduled for each month until all episodes were collected into eight volumes. Each volume contains 3 episodes. A DVD Box Set of the first season of Index was released on December 22, 2010 containing the first twelve episodes of the season, with a second Box Set scheduled to be released on March 9, 2011 containing the rest of the season. The first Blu-ray and DVD compilation of the second season of Index was released on January 26, 2011, and followed the same model as the first season's Blu-ray and DVD release schedule. | END ID: 216

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 217 | TITLE: Tangent lines to circles | CONTENT: Another method to construct the tangent lines to a point P external to the circle using only a straightedge: | END ID: 217

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 218 | TITLE: History of Tibet | CONTENT: The community of Tibetans in exile established in Dharamsala and Bylakuppe near Mysore in Karnataka, South India, has expanded since 1959. Tibetans have duplicated Tibetan monasteries in India and these now house tens of thousands of monks. They have also created Tibetan schools and hospitals, and founded the Library of Tibetan Works and Archives — all aimed at continuing Tibetan tradition and culture. Tibetan festivals such as Lama dances, celebration of Losar (the Tibetan New Year), and the Monlam Prayer Festival, continue in exile. | END ID: 218

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 219 | TITLE: Joseph Beuys | CONTENT: Franz Joseph and Hans van der Grinten organized Beuys' first solo show at their house in Kranenburg in 1953. The Alfred Schmela Galerie was the first commercial gallery to hold a Beuys solo exhibition in 1965. Beuys participated for the first time in Documenta in Kassel in 1964. In 1969, he was included in Harald Szeemann's groundbreaking exhibition When Attitudes Become Form at the Kunsthalle Bern. | END ID: 219

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 220 | TITLE: List of historical acts of tax resistance | CONTENT: In 1903, tens of thousands of British nonconformists began resisting the part of their taxes that paid for sectarian schools. Over 170 would eventually be jailed for their tax refusal.[170] | END ID: 220

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 221 | TITLE: Academy Award for Best Picture | CONTENT: Several musical adaptations based on material previously filmed in non-musical form have won Best Picture, including Gigi, West Side Story, My Fair Lady, The Sound of Music, Oliver!, and Chicago. | END ID: 221

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 222 | TITLE: List of The 100 episodes | CONTENT: As of May 24, 2017,[update] 58 episodes of The 100 have aired, concluding the fourth season. In March 2017, The CW renewed the series for a fifth season, set to premiere on April 24, 2018.[7][8] | END ID: 222

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 223 | TITLE: Salt Lake City | CONTENT: KTVX 4 signed on the air as Utah's first television station in 1947 under the experimental callsign W6SIX. KTVX is the oldest TV station in the Mountain Time Zone and the third oldest west of the Mississippi. It is Salt Lake City's current ABC affiliate. KSL-TV 5, the local NBC affiliate, has downtown studios at "Broadcast House" in the Triad Center office complex. KSL is operated by Deseret Media Companies, a company owned by the LDS Church. KUTV 2 is Salt Lake City's CBS affiliate. KSTU 13 is the area's Fox affiliate. KUCW 30 is the CW affiliate and part of a duopoly with KTVX. KJZZ-TV 14 is an independent station owned by Sinclair Broadcast Group, and is part of a triopoly with KUTV and St. George-licensed MyNetworkTV affiliate KMYU 12. | END ID: 223

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 224 | TITLE: Voting behavior | CONTENT: Voting behavior is a form of electoral behavior. Understanding voters' behavior can explain how and why decisions were made either by public decision-makers, which has been a central concern for political scientists,[1] or by the electorate. To interpret voting behavior both political science and psychology expertise were necessary and therefore the field of political psychology emerged. Political psychology researchers study ways in which affective influence may help voters make more informed voting choices, with some proposing that affect may explain how the electorate makes informed political choices in spite of low overall levels of political attentiveness and sophistication. | END ID: 224

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 225 | TITLE: List of Trailer Park Boys characters | CONTENT: At this time, Lahey claimed that Ricky was his biological son, explaining that this caused Ricky's mother to leave Sunnyvale and Jim and Ray to hate each other. However, in Season Eleven, after Ricky discovers the "truth" of his heritage, it is revealed that Lahey is not his father, although Lahey himself firmly believed that he was. After doing research at the hospital, Bubbles and Julian discovered that neither Lahey nor Ray are Ricky's biological father, but they continue to let him believe that Ray is his real father, as he is the man who raised him. | END ID: 225

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 226 | TITLE: Ellen Churchill Semple | CONTENT: Ellen Churchill Semple (January 8, 1863 â€“ May 8, 1932) was an American geographer and the first female president of the Association of American Geographers. She contributed significantly to the early development of the discipline of geography in the United States, particularly studies of human geography. She is most closely associated with work in anthropogeography and environmentalism, and the debate about "environmental determinism". | END ID: 226

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 227 | TITLE: Pluto | CONTENT: From 1992 onward, many bodies were discovered orbiting in the same volume as Pluto, showing that Pluto is part of a population of objects called the Kuiper belt. This made its official status as a planet controversial, with many questioning whether Pluto should be considered together with or separately from its surrounding population. Museum and planetarium directors occasionally created controversy by omitting Pluto from planetary models of the Solar System. The Hayden Planetarium reopened—in February 2000, after renovation—with a model of only eight planets, which made headlines almost a year later.[45] | END ID: 227

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 228 | TITLE: Aftermath of World War I | CONTENT: Through the period from the armistice on 11 November 1918 until the signing of the peace treaty with Germany on 28 June 1919, the Allies maintained the naval blockade of Germany that had begun during the war. As Germany was dependent on imports, it is estimated that 523,000 civilians had lost their lives.[1] N. P. Howard, of the University of Sheffield, claims that a further quarter of a million more died from disease or starvation in the eight-month period following the conclusion of the conflict.[2] The continuation of the blockade after the fighting ended, as author Robert Leckie wrote in Delivered From Evil, did much to "torment the Germans ... driving them with the fury of despair into the arms of the devil."[citation needed] The terms of the Armistice did allow food to be shipped into Germany, but the Allies required that Germany provide the means (the shipping) to do so. The German government was required to use its gold reserves, being unable to secure a loan from the United States.[citation needed] | END ID: 228

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 229 | TITLE: Agents of S.H.I.E.L.D. (season 5) | CONTENT: The fifth season is set to begin airing on December 1, 2017, after Marvel's Inhumans has finished airing its episodes, and run for 22 episodes. | END ID: 229

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 230 | TITLE: Fibula | CONTENT: The fibula or calf bone is a leg bone located on the lateral side of the tibia, with which it is connected above and below. It is the smaller of the two bones, and, in proportion to its length, the slenderest of all the long bones. Its upper extremity is small, placed toward the back of the head of the tibia, below the level of the knee joint, and excluded from the formation of this joint. Its lower extremity inclines a little forward, so as to be on a plane anterior to that of the upper end; it projects below the tibia, and forms the lateral part of the ankle-joint. | END ID: 230

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 231 | TITLE: J Balvin | CONTENT: In June 2015, it was announced that J Balvin had cancelled his performance on Miss USA 2015 to protest Donald Trump's inflammatory comments insulting illegal immigrants,[3][2][4] saying, "During [Trump's] presidential campaign kickoff speech last week [June 2015], Trump accused illegal immigrants of bringing drugs, crime and rapists to the U.S."[5][6] His live performance had been scheduled for July 12, 2015 in Louisiana, which would have been J Balvin's first performance on national mainstream television.[7] J Balvin also began the #LatinosSomosUnidos movement through social media, to bring awareness to the struggles occurring on the Venezuelan-Colombian border. His campaign brought support from many of the major Latin artists. | END ID: 231

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 232 | TITLE: Auspicious wedding date | CONTENT: A February bride will be an affectionate wife, And a tender mother. | END ID: 232

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 233 | TITLE: Carmen Sandiego (video game series) | CONTENT: Producer: Brøderbund Software Inc. Publisher: Brøderbund Software Inc. Year: 1991 | END ID: 233

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 234 | TITLE: 2016–17 Chelsea F.C. season | CONTENT: On 22 January, The Blues defeated Hull City 2-0 at home. Diego Costa scored at his 100th appearance for the club at the 7th minute of first-half injury time. The long stoppage was a result of a clash of heads with between Gary Cahill and Hull midfielder Ryan Mason. Mason was sent to hospital and it was later confirmed that he sustained a skull fracture, while Cahill remained on the pitch and secured the victory with a header goal on the second half.[115] | END ID: 234

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 235 | TITLE: Cinque Ports | CONTENT: Exemption from tax and tallage, right of soc and sac, tol and team, blodwit (the right to punish shedders of blood) and fledwit (the right to punish those who were seized in an attempt to escape from justice), pillory and tumbril, infangentheof and outfangentheof, mundbryce (the breaking into or violation of a man's mund or property in order to erect banks or dikes as a defence against the sea), waifs and strays, flotsam and jetsam and ligan | END ID: 235

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 236 | TITLE: Crime in the United States | CONTENT: According to the FBI, "When the race of the offender was known, 53.0 percent were black, 44.7 percent were white, and 2.3 percent were of other races. The race was unknown for 4,132 offenders. (Based on Expanded Homicide Data Table 3). Of the offenders for whom gender was known, 88.2 percent were male."[54] According to the U.S. Bureau of Justice Statistics, from 1980 to 2008, 84 percent of white homicide victims were killed by white offenders and 93 percent of black homicide victims were killed by black offenders.[29] | END ID: 236

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 237 | TITLE: Love Never Dies (musical) | CONTENT: On 3 July 2009, Lloyd Webber announced that Karimloo (who had played the Phantom in the West End) and Sierra Boggess (who had originated the role of Christine in Phantom â€“ The Las Vegas Spectacular) had been cast as the Phantom and Christine and that the role of Meg Giry would be played by Summer Strallen, Madame Giry by Liz Robertson, and Raoul by Joseph Millson.[22][23] I'd Do Anything finalist Niamh Perry was given the role of Fleck.[24] | END ID: 237

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 238 | TITLE: Diocletian | CONTENT: After Diocletian's reform of the provinces, governors were called iudex, or judge. The governor became responsible for his decisions first to his immediate superiors, as well as to the more distant office of the emperor.[250] It was most likely at this time that judicial records became verbatim accounts of what was said in trial, making it easier to determine bias or improper conduct on the part of the governor. With these records and the Empire's universal right of appeal, Imperial authorities probably had a great deal of power to enforce behavior standards for their judges.[251] In spite of Diocletian's attempts at reform, the provincial restructuring was far from clear, especially when citizens appealed the decisions of their governors. Proconsuls, for example, were often both judges of first instance and appeal, and the governors of some provinces took appellant cases from their neighbors. It soon became impossible to avoid taking some cases to the emperor for arbitration and judgment.[252] Diocletian's reign marks the end of the classical period of Roman law. Where Diocletian's system of rescripts shows an adherence to classical tradition, Constantine's law is full of Greek and eastern influences.[253] | END ID: 238

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 239 | TITLE: Jugantar Patrika | CONTENT: The paper rapidly acquired a broad popularity, at one time having a readership of 20,000. Bhupendranath Dutt served as the editor of the newspaper till his arrest in 1907, although it also published articles from a number of noted Bengali revolutionaries including Barindra Kumar Ghosh and Aurobindo Ghosh. It faced prosecution a number of times by the British Indian government for publishing seditious articles. Bhupendranath Dutt was arrested in 1907 for publication of articles "inciting violence against the Government of India", for which he was sentenced to a year's rigorous imprisonment. The paper was ultimately forced to shut down in 1908, amidst financial ruins following the prosecutions, and after the passage of The Newspapers (Incitement to offences) act in June 1908 which made its position vulnerable. | END ID: 239

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 240 | TITLE: Western canon | CONTENT: Världsbiblioteket (The World Library) was a Swedish list of the 100 best books in the world, created in 1991 by the Swedish literary magazine Tidningen Boken. The list was compiled through votes from members of the Svenska Akademien, Swedish Crime Writers' Academy, librarians, authors, and others. Approximately 30 of the books were Swedish. | END ID: 240

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 241 | TITLE: Transistor | CONTENT: The key advantages that have allowed transistors to replace vacuum tubes in most applications are | END ID: 241

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 242 | TITLE: Sufism | CONTENT: In short, Muslim scholars who focused their energies on understanding the normative guidelines for the body came to be known as jurists, and those who held that the most important task was to train the mind in achieving correct understanding came to be divided into three main schools of thought: theology, philosophy, and Sufism. This leaves us with the third domain of human existence, the spirit. Most Muslims who devoted their major efforts to developing the spiritual dimensions of the human person came to be known as Sufis.[24] | END ID: 242

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 243 | TITLE: List of Law & Order: Special Victims Unit characters | CONTENT: As for Rollins' personal life, little is mentioned of her off-duty life (although, being from Atlanta, she is a fan of the Braves, whose schedule she keeps on her refrigerator door); Amanda has mentioned that she has a sister, Kim, who has had psychotic and drug issues. Kim has also suffered repeated abuse by her ex-boyfriend.[10] She says that while she was working in Atlanta, there was an accident that occurred that allowed for her to transfer to the SVU.[11] Amanda also was exposed as a heavy gambler in the episode "Home Invasions". When Cragen discovered her problem, he threatened to take her badge, but decided to help instead—since he is a recovering alcoholic—by requiring her to attend Gamblers Anonymous meetings.[12] Rollins' previously mentioned troubled sister, Kim (Lindsay Pulsipher), comes to New York in the season 14 episode, "Friending Emily", causing problems for Amanda while she is trying to work a case. Later in the episode "Deadly Ambition", Kim returns to New York beaten by her ex-boyfriend Jeff and claiming to be pregnant. When Amanda hears screams from inside her apartment, she finds Kim's ex-boyfriend attempting to rape Kim, and Amanda shoots and kills the man as he pulls a gun on her. The supposed evidence of Amanda shooting Jeff in cold blood leads to Lt. Tucker arresting Amanda in Captain Cragen's office. The charges against Amanda are later dropped when Amaro tapes Kim confessing to setting Amanda up for a life insurance policy on Jeff. Before Kim can be arrested, however, she steals everything from Amanda's apartment and disappears. In the episode "Poisoned Motive", Rollins is shot by a sniper in front of the precinct. Her shooting leads back to the daughter of Detective Tutuola's narcotics partner, who is out for revenge on the NYPD after her father was injured on the job by protecting Tutuola from a bullet. | END ID: 243

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 244 | TITLE: Large intestine | CONTENT: The transverse colon is encased in peritoneum, and is therefore mobile (unlike the parts of the colon immediately before and after it). | END ID: 244

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 245 | TITLE: Cabela's | CONTENT: Cabela's Inc. is an American direct marketer and specialty retailer of hunting, fishing, boating, camping, shooting, and related outdoor recreation merchandise, based in Sidney, Nebraska. The company was founded by Richard N. Cabela in 1961 and went public in 2004, with that fiscal year's revenue reaching $1.56 billion, a 50% growth since 2001. | END ID: 245

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 246 | TITLE: Calendar | CONTENT: Because the number of days in the tropical year is not a whole number, a solar calendar must have a different number of days in different years. This may be handled, for example, by adding an extra day in leap years. The same applies to months in a lunar calendar and also the number of months in a year in a lunisolar calendar. This is generally known as intercalation. Even if a calendar is solar, but not lunar, the year cannot be divided entirely into months that never vary in length. | END ID: 246

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 247 | TITLE: Palmer Raids | CONTENT: In June 1919, Attorney General Palmer told the House Appropriations Committee that all evidence promised that radicals would "on a certain day...rise up and destroy the government at one fell swoop." He requested an increase in his budget to $2,000,000 from $1,500,000 to support his investigations of radicals, but Congress limited the increase to $100,000.[5] | END ID: 247

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 248 | TITLE: Dakota Territory | CONTENT: Dakota Territory is the main setting for the HBO TV series "Deadwood". The town of Yankton and the Black Hills area are mentioned often in the show. | END ID: 248

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 249 | TITLE: Grade inflation | CONTENT: A January 7, 2009 article in the Pittsburgh Post-Gazette used the term "grade inflation" to describe how some people viewed a grading policy in the Pittsburgh public school district. According to the article, the policy sets 50% as the minimum score that a student can get on any given school assignment. The article also stated that some students said they would rather get a score of 50% than do the school work.[27] A March 2, 2009 follow-up article in the same newspaper said that the policy had been amended so that students who refuse to do the work will receive a grade of zero, and that the minimum grade of 50% will only apply to students who make a "good-faith effort".[28] A March 3, 2009, article in the same newspaper quoted Bill Hileman, a Pittsburgh Federation of Teachers staff representative, as saying, "The No. 1 problem with the 50 percent minimum was the negative impact on student behavior." The same article also said that the school district was planning to adopt a new grading scale in at least two schools by the end of the month. The article stated that under the original grading scale, the minimum scores required to earn an A, B, C, D, or F, were, respectively, 90%, 80%, 70%, 60%, and 0%. Under the new 5-point grading scale, the minimum scores required to earn an A, B, C, D, or F would be changed, respectively, to 4.0, 3.0, 2.0, 1.0, and 0.[29] | END ID: 249

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 250 | TITLE: Approximations of π | CONTENT: Advances in the approximation of π (when the methods are known) were made by increasing the number of sides of the polygons used in the computation. A trigonometric improvement by Willebrord Snell (1621) obtains better bounds from a pair of bounds gotten from the polygon method. Thus, more accurate results were obtained from polygons with fewer sides.[47] Viète's formula, published by François Viète in 1593, was derived by Viète using a closely related polygonal method, but with areas rather than perimeters of polygons whose numbers of sides are powers of two.[48] | END ID: 250

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 251 | TITLE: Geoffrey Palmer (actor) | CONTENT: Geoffrey Dyson Palmer, OBE (born 4 June 1927) is an English actor known for his roles in British television sitcoms playing Jimmy Anderson in The Fall and Rise of Reginald Perrin (1976–79), Ben Parkinson in Butterflies (1978–83) and Lionel Hardcastle in As Time Goes By (1992–2005). His film appearances include A Fish Called Wanda (1988), The Madness of King George (1994), Mrs. Brown (1997), and Tomorrow Never Dies (1997). | END ID: 251

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 252 | TITLE: Glee (season 4) | CONTENT: The fourth season of the Fox musical comedy-drama television series Glee was commissioned on April 9, 2012.[1][2] It premiered on September 13, 2012 and is produced by 20th Century Fox Television, Ryan Murphy Television and Brad Falchuk Teley-Vision with executive producers Dante Di Loreto and series co-creators Ryan Murphy, Brad Falchuk and Ian Brennan.[3] | END ID: 252

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 253 | TITLE: Energy policy of Canada | CONTENT: Canada has a robust energy profile with abundant and diverse resources. Energy and climate policies are interrelated. These policies are implemented at both the federal and provincial governmental level. A recent SWOT (Strengths, Weaknesses, Opportunities, and Threats) analysis conducted in 2013 of a Canadian energy and climate policies has shows that there is a lack of consistency between federal and regional strategies.[7] The reason for this lack of consistency was attributed to the economic and environmental realities, the diversity of energy sources and energy demands that vary greatly among the Canadian provinces. As a result of the differing energy characteristics of the provinces there is creation of multiple federal and provincial strategies, sometimes complementary, but often contradictory. | END ID: 253

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 254 | TITLE: San Antonio Spurs | CONTENT: Despite the shortened 66-game NBA season due to the NBA lockout, the Spurs won 50 games and tied the Chicago Bulls for the best record in the league. They extended their streak of 50+ win seasons to 13 since the 1999–2000 season, an NBA record. Popovich won his second Coach of the Year.[37] | END ID: 254

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 255 | TITLE: A (Pretty Little Liars) | CONTENT: After the revelation of Mona Vanderwaal as the first and original "A", she began receiving visits from someone, known as Red Coat, who offered her a partnership and together they built up the "A-Team". The team had many members but disbanded after the season three finale and Big A began working with a single ally. The identity of the second "A", Red Coat, and the leader of the "A-Team" was revealed to be CeCe Drake, while her ally that donned the Black Widow and other Red Coat disguise was revealed to be Sara Harvey. Five years later, a new mysterious entity arises and begins using Emojis to communicate but later baptizes themselves as "A.D.", while the Liars refer to the anonymous figure as Uber A. Then, in the Series Finale, "A.D." reveals themselves to be Alex Drake, the twin sister of Spencer. | END ID: 255

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 256 | TITLE: Lunar effect | CONTENT: Three studies carried out between 1959 and 1973 reported a 1 percent increase in births in New York following a full Moon.[citation needed] However, multiple studies have found no connection between birth rate and lunar phases. A 1957 analysis of 9,551 births in Danville, PA, found no correlation between birth rate and the phase of the Moon.[9] Records of 11,961 live births and 8,142 natural births (not induced by drugs or cesarean section) over a 4-year period (1974-1978) at the UCLA hospital did not correlate in any way with the cycle of lunar phases.[10] Analysis of 3,706 spontaneous births (excluding births resulting from induced labor) in 1994 showed no correlation with lunar phase.[11] The distribution of 167,956 spontaneous vaginal deliveries, at 37 to 40 weeks gestation, in Phoenix, AZ, between 1995 and 2000, showed no relationship with lunar phase.[12] Analysis of 564,039 births (1997 to 2001) in North Carolina showed no predictable influence of the lunar cycle on deliveries or complications.[13] Analysis of 6,725 deliveries (2000 to 2006) in Hannover revealed no significant correlation of birth rate to lunar phases.[14] A 2001 analysis of 70,000,000 birth records from the National Center for Health Statistics revealed no correlation between birth rate and lunar phase.[15] An extensive review of 21 studies from 7 different countries showed that the majority of studies reported no relationship to lunar phase, and that the positive studies were inconsistent with each other.[2] A review of 6 additional studies from 5 different countries similarly showed no evidence of relationship between birth rate and lunar phase.[16] | END ID: 256

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 257 | TITLE: Anna (Frozen) | CONTENT: While there hadn't been any official announcements from Disney regarding a coronation for Anna and Elsa, it had been announced in late August 2014 that a special character meal would be held by a group of travel agents in the morning of September 24, 2014. While not officially organized by Disney, the event, called My Royal Coronation, would feature the official Anna and Elsa characters owned by Disney with assistance from the company.[90] On September 12, 2014, Walt Disney World announced that a Frozen attraction was scheduled to open in early 2016 at Epcot's World Showcase in the Norway pavilion, replacing the park's Maelstrom ride. The attraction features the kingdom of Arendelle with music and scenes from the film, as well as meet-and-greets with Anna and Elsa.[91][92] Anna, Elsa, Kristoff, and Olaf will make appearances in Mickey’s Once Upon a Christmastime Parade, offered during Mickey’s Very Merry Christmas Party at Magic Kingdom in November and December 2014[91] (from November 7 to December 31).[92] | END ID: 257

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 258 | TITLE: Intelligent design | CONTENT: Those who disagree with our holding will likely mark it as the product of an activist judge. If so, they will have erred as this is manifestly not an activist Court. | END ID: 258

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 259 | TITLE: Portland, Maine | CONTENT: The city is home to three minor league teams. The Portland Sea Dogs, the Double-A farm team of the Boston Red Sox, play at Hadlock Field. The Maine Red Claws, the NBA G League affiliate of the Boston Celtics, play at the Portland Exposition Building. The GPS Portland Phoenix soccer teams plays in the Premier Development League. | END ID: 259

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 260 | TITLE: Validity (statistics) | CONTENT: Face validity is an estimate of whether a test appears to measure a certain criterion; it does not guarantee that the test actually measures phenomena in that domain. Measures may have high validity, but when the test does not appear to be measuring what it is, it has low face validity. Indeed, when a test is subject to faking (malingering), low face validity might make the test more valid. Considering one may get more honest answers with lower face validity, it is sometimes important to make it appear as though there is low face validity whilst administering the measures. | END ID: 260

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 261 | TITLE: Australian contract law | CONTENT: Where there is no time is specified for performance, the law implies an obligation to perform within a reasonable time.[181][182] In such circumstances, it is highly unlikely that time will be viewed as being "of the essence",[179] unless failure to perform within a reasonable time will have serious consequences for the aggrieved party.[183] | END ID: 261

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 262 | TITLE: Land Rover Discovery | CONTENT: The vehicle was lauded by the press, with the Terrain Response system, improved on-road dynamics, and interior design receiving particular praise. Jeremy Clarkson of the BBC's Top Gear motoring show drove one to the top of Cnoc an Fhreiceadain, a 307 m (1,007 ft) mountain near Tongue in northern Scotland, where no vehicle had previously reached. Richard Hammond, presenter of Top Gear, praised it as the "Best 4X4 of all time". In Australia, the vehicle was awarded "4WD of the Year" by the 4WD Press. | END ID: 262

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 263 | TITLE: Museum of Fine Arts, Boston | CONTENT: Founded in 1870, the museum moved to its current location in 1909. The museum is affiliated with the School of the Museum of Fine Arts at Tufts. | END ID: 263

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 264 | TITLE: Central Powers | CONTENT: United Baltic Duchy | END ID: 264

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 265 | TITLE: Brighton Palace Pier | CONTENT: In 2004, the Brighton Marine Palace Pier Company (owned by the Noble Organisation), admitted an offence of breaching public safety under the Health and Safety at Work Act and had to pay fines and costs of Â£37,000 after a fairground ride was operated with part of its track missing. A representative from the Health and Safety Executive said that inadequate procedures were to blame for the fact that nothing had been done to alert staff or passengers that the ride would be dangerous to use.[22] The pier management came into criticism from Brighton and Hove City Council, who thought they were relying too much on fairground rides, some of which were being built too high.[9] | END ID: 265

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 266 | TITLE: Han dynasty | CONTENT: The most common occupation for women was weaving clothes for the family, sale at market or for large textile enterprises that employed hundreds of women. Other women helped on their brothers' farms or became singers, dancers, sorceresses, respected medical physicians, and successful merchants who could afford their own silk clothes.[128] Some women formed spinning collectives, aggregating the resources of several different families.[129] | END ID: 266

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 267 | TITLE: Costa Concordia disaster | CONTENT: Costa Concordia (call sign: IBHD, IMO number: 9320544, MMSI number: 247158500), with 3,206 passengers and 1,023 crew members on board,[2] was sailing off Isola del Giglio on the night of 13 January 2012, having begun a planned seven-day cruise from Civitavecchia, Lazio, Italy, to Savona and five other ports.[21] She struck her port side on a reef,[22][23] at 21:42 or 21:45 local time.[24] The reef is charted as an area known as Le Scole,[25][26] about 800 metres (870Â yd) south of the entrance to the harbour of Giglio Porto, on the island's east coast. | END ID: 267

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 268 | TITLE: Military police | CONTENT: The word can have different meanings in different countries, and may refer to: | END ID: 268

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 269 | TITLE: Serous membrane | CONTENT: In anatomy, serous membrane (or serosa) is a smooth tissue membrane consisting of two layers of mesothelium, which secrete serous fluid. The inner layer that covers organs (viscera) in body cavities is called the visceral membrane. A second layer of epithelial cells of the serous membrane, called the parietal layer, lines the body wall. Between the two layers is a potential space, mostly empty except for a few milliliters of lubricating serous fluid that is secreted by the two serous membranes.[1] | END ID: 269

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 270 | TITLE: Over the Garden Wall | CONTENT: The series follows two half-brothers, Wirt and Greg (voiced by Elijah Wood and Collin Dean respectively), who become lost in a strange forest called the Unknown. In order to find their way home, the two must travel across the seemingly supernatural forest with the occasional help of the wandering, mysterious and elderly Woodsman (Christopher Lloyd) and Beatrice (Melanie Lynskey), an irritable bluebird who travels with the boys in order to find a woman called Adelaide, who can supposedly undo the curse on Beatrice and her family and show the half-brothers the way home.[1] | END ID: 270

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 271 | TITLE: Islam in the United Kingdom | CONTENT: A 2009 government paper estimated the Nigerian Muslim community as 12,000 to 14,000.[74] The community is concentrated in London. | END ID: 271

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 272 | TITLE: Serbia | CONTENT: Law enforcement is the responsibility of the Serbian Police, which is subordinate to the Ministry of the Interior. Serbian Police fields 26,527 uniformed officers.[120] National security and counterintelligence are the responsibility of the Security Intelligence Agency (BIA).[121] | END ID: 272

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 273 | TITLE: G. Sankara Kurup | CONTENT: G. Sankara Kurup, (3 June 1901, Nayathode, Kingdom of Cochin (now in Ernakulam district, Kerala, India) â€“ 2 February 1978, Vappalassery, Angamaly, Ernakulam district, Kerala), better known as Mahakavi G (The Great Poet G), was the first winner of the Jnanpith Award, India's highest literary award.[1][2] He won the prize in 1965 for his collection of poems in Malayalam Odakkuzhal (The Bamboo Flute, 1950). With part of the prize money he established the literary award Odakkuzhal in 1968. He was also the recipient of the Soviet Land Nehru Award, in 1967, and the Padma Bhushan in 1968.[3] His poetry collection Viswadarshanam won the Kerala Sahitya Akademi Award in 1961 and Kendra Sahitya Akademi Award in 1963. | END ID: 273

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 274 | TITLE: LoveGame | CONTENT: The video starts with the heading "Streamline presents" and three men moving through Times Square. They open a manhole cover on which "Haus of Gaga" is written. Gaga is then shown naked with blue and purple paint and glitter on her body, frolicking with two men who have the words "Love" and "Fame" shaved into their heads. The scene shifts to a subway where Gaga starts singing in a grey-white leotard with a hood. She carries her characteristic disco stick and wears chain-linked glasses. The chorus starts with Gaga and her dancers progressing through the subway and dancing down a staircase. Two harlequin Great Danes, are also shown on top of the staircase.[52] | END ID: 274

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 275 | TITLE: United States federal budget | CONTENT: The "extended baseline scenario" assumes that the laws currently on the books will be implemented, for the most part. CBO reported in July 2014 that under this scenario: "If current laws remained generally unchanged in the future, federal debt held by the public would decline slightly relative to GDP over the next few years. After that, however, growing budget deficits would push debt back to and above its current high level. Twenty-five years from now, in 2039, federal debt held by the public would exceed 100 percent of GDP. Moreover, debt would be on an upward path relative to the size of the economy, a trend that could not be sustained indefinitely. By 2039, the deficit would equal 6.5 percent of GDP, larger than in any year between 1947 and 2008, and federal debt held by the public would reach 106 percent of GDP, more than in any year except 1946â€”even without factoring in the economic effects of growing debt."[13] | END ID: 275

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 276 | TITLE: Morse code | CONTENT: In the United Kingdom, many people learned the Morse code by means of a series of words or phrases that have the same rhythm as a Morse character. For instance, "Q" in Morse is dah-dah-di-dah, which can be memorized by the phrase "God save the Queen", and the Morse for "F" is di-di-dah-dit, which can be memorized as "Did she like it." | END ID: 276

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 277 | TITLE: Criticism of Franklin D. Roosevelt | CONTENT: Both during and after his presidential terms and continuing today, there has been much criticism of Franklin D. Roosevelt. Critics have questioned not only his policies and positions, but also charged him with centralizing power in his own hands by controlling both the government and the Democratic Party. Many denounced his breaking the no-third-term tradition in 1940.[1] | END ID: 277

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 278 | TITLE: Edinburgh Festival Fringe | CONTENT: The Edinburgh Festival Fringe (often referred to as simply The Fringe) is the world's largest arts festival, which in 2017 spanned 25 days and featured 53,232 performances of 3,398 shows[1] in 300 venues.[2] Established in 1947 as an alternative to the Edinburgh International Festival, it takes place annually in Edinburgh, Scotland, in the month of August.[3] | END ID: 278

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 279 | TITLE: Final Five (gymnastics) | CONTENT: In the balance beam final, Hernandez won the silver medal, finishing behind the Netherland's Sanne Wevers. Biles suffered a shocking mishap when she put her hands on the beam after a balance check on her front tuck. Despite the mistake, her score was high enough to win her the bronze medal. | END ID: 279

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 280 | TITLE: Civil forfeiture in the United States | CONTENT: Courts helped set up the legal framework to help law enforcement stem the drug tide while sometimes trying to rein in abuses. A 1984 law set up the equitable sharing arrangement in which state and local police can share the seizures with federal agents.[14] While the 1993 Supreme Court case Austin v. United States ruled that a forfeiture could be considered as an excessive fine,[15] the court upheld the principle of civil forfeiture generally.[7] A 1996 Supreme Court decision ruled that prosecuting a person for a crime and seizing his or her property via civil forfeiture did not constitute double jeopardy, and therefore did not violate the Constitution.[15] However, in 1999, the Supreme Court ruled that civil forfeiture was not permitted if the amount seized was "grossly disproportional" to the gravity of the offense.[6] | END ID: 280

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 281 | TITLE: Marrakesh | CONTENT: Metalwork made in Marrakesh includes brass lamps, iron lanterns, candle holders made from recycled sardine tins, and engraved brass teapots and tea trays used in the traditional serving of tea. Contemporary art includes sculpture and figurative paintings. Blue veiled Tuareg figurines and calligraphy paintings are also popular.[141] | END ID: 281

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 282 | TITLE: Heysel Stadium disaster | CONTENT: During Euro 2000, members of the Italian team left flowers on the site, in honour of the victims. | END ID: 282

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 283 | TITLE: Bald eagle | CONTENT: The bald eagle placed in the genus Haliaeetus (sea eagles) which gets both its common and specific scientific names from the distinctive appearance of the adult's head. Bald in the English name is derived from the word piebald, and refers to the white head and tail feathers and their contrast with the darker body.[18] The scientific name is derived from Haliaeetus, New Latin for "sea eagle" (from the Ancient Greek haliaetos), and leucocephalus, Latinized Ancient Greek for "white head," from λευκος leukos ("white") and κεφαλη kephale ("head").[19][20] | END ID: 283

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 284 | TITLE: List of Justice League episodes | CONTENT: Following the death of the Flash, the Justice Lords launch an assault on the White House, where Superman kills President Lex Luthor. Two years later, the Lords now rule over the planet with an iron fist. Batman discovers the dimension which the Justice League inhabits. Considering their counterparts naive, but wishing to spread order to the newly discovered world, they cross over and trap the League in a force field. They then take their places in a quest to make this Earth like their own. | END ID: 284

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 285 | TITLE: No. 617 Squadron RAF | CONTENT: The Squadron reformed on 1 January 1983 at RAF Marham, re-equipped with twelve Tornado GR1 aircraft and eighteen WE.177 nuclear bombs,[21] and the Squadron's role assigned to SACEUR remained one of support for land forces on the Continent. Its Tornado aircraft were each able to carry two WE.177 bombs and the ratio of weapons to aircraft at full strength increased to 1.5:1, with an allowance now made for attrition in the conventional opening phase of a continental war. The Squadron continued in this role until the WE.177 weapons were retired and No. 617 Squadron relinquished its nuclear delivery capability.[22] | END ID: 285

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 286 | TITLE: Egyptian hieroglyphs | CONTENT: Egyptian hieroglyphs (/ˈhaɪrəˌɡlɪf, -roʊ-/[2][3]) were the formal writing system used in Ancient Egypt. It combined logographic, syllabic and alphabetic elements, with a total of some 1,000 distinct characters.[4][5] Cursive hieroglyphs were used for religious literature on papyrus and wood. The later hieratic and demotic Egyptian scripts were derived from hieroglyphic writing; Meroitic was a late derivation from demotic. | END ID: 286

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 287 | TITLE: The Hero: Love Story of a Spy | CONTENT: The music is composed by Uttam Singh. Lyrics are penned by Anand Bakshi and Javed Akhtar. | END ID: 287

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 288 | TITLE: x86-64 | CONTENT: Long mode is the architecture's intended primary mode of operation; it is a combination of the processor's native 64-bit mode and a combined 32-bit and 16-bit compatibility mode. It is used by 64-bit operating systems. Under a 64-bit operating system, 64-bit programs run under 64-bit mode, and 32-bit and 16-bit protected mode applications (that do not need to use either real mode or virtual 8086 mode in order to execute at any time) run under compatibility mode. Real-mode programs and programs that use virtual 8086 mode at any time cannot be run in long mode unless those modes are emulated in software.[11]:11 However, such programs may be started from an operating system running in long mode on processors supporting VT-x or AMD-V by creating a virtual processor running in the desired mode. | END ID: 288

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 289 | TITLE: History of tea | CONTENT: In one popular Chinese legend, Shennong, the legendary Emperor of China and inventor of agriculture and Chinese medicine was drinking a bowl of just boiled water due to a decree that his subjects must boil water before drinking it [10] some time around 2737 BC when a few leaves were blown from a nearby tree into his water, changing the color. The emperor took a sip of the brew and was pleasantly surprised by its flavor and restorative properties. A variant of the legend tells that the emperor tested the medical properties of various herbs on himself, some of them poisonous, and found tea to work as an antidote.[11] Shennong is also mentioned in Lu Yu's famous early work on the subject, The Classic of Tea.[12] A similar Chinese legend goes that the god of agriculture would chew the leaves, stems, and roots of various plants to discover medicinal herbs. If he consumed a poisonous plant, he would chew tea leaves to counteract the poison. | END ID: 289

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 290 | TITLE: Human rights in Nigeria | CONTENT: Throughout Nigeria, religious minorities are systematically restricted from building places of worship and schools through the denial of land grants.  Members of minority religious groups are often attacked during riots and religious conflicts.[34] | END ID: 290

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 291 | TITLE: Paul the Apostle | CONTENT: F. C. Baur (1792–1860), professor of theology at Tübingen in Germany, the first scholar to critique Acts and the Pauline Epistles, and founder of the Tübingen School of theology, argued that Paul, as the "Apostle to the Gentiles", was in violent opposition to the original 12 Apostles. Baur considers the Acts of the Apostles were late and unreliable. This debate has continued ever since, with Adolf Deissmann (1866–1937) and Richard Reitzenstein (1861–1931) emphasising Paul's Greek inheritance and Albert Schweitzer stressing his dependence on Judaism. | END ID: 291

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 292 | TITLE: Physical layer | CONTENT: In the seven-layer OSI model of computer networking, the physical layer or layer 1 is the first and lowest layer.[1] This layer may be implemented by a PHY chip. | END ID: 292

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 293 | TITLE: Chromosome | CONTENT: Chromosomes are normally visible under a light microscope only when the cell is undergoing the metaphase of cell division (where all chromosomes are aligned in the center of the cell in their condensed form).[3] Before this happens, every chromosome is copied once (S phase), and the copy is joined to the original by a centromere, resulting either in an X-shaped structure (pictured to the right) if the centromere is located in the middle of the chromosome or a two-arm structure if the centromere is located near one of the ends. The original chromosome and the copy are now called sister chromatids. During metaphase the X-shape structure is called a metaphase chromosome. In this highly condensed form chromosomes are easiest to distinguish and study.[4] In animal cells, chromosomes reach their highest compaction level in anaphase during segregation.[5] | END ID: 293

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 294 | TITLE: Think different | CONTENT: Similar portraits were also posted without the "Think different" text on at least seven additional occasions: | END ID: 294

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 295 | TITLE: Encephalitis | CONTENT: Certain types are preventable with vaccines.[5] Treatment may include, antiviral medication (such as acyclovir), anticonvulsants, and corticosteroids.[1] Treatment generally takes place in hospital.[1] Some people require artificial respiration.[1] Once the immediate problem is under control, rehabilitation may be required.[2] In 2015, encephalitis was estimated to have affected 4.3 million people and resulted in 150,000 deaths worldwide.[3][4] | END ID: 295

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 296 | TITLE: Disinfectant | CONTENT: An alternative assessment is to measure the Minimum inhibitory concentrations (MICs) of disinfectants against selected (and representative) microbial species, such as through the use of microbroth dilution testing.[16] | END ID: 296

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 297 | TITLE: Chetan Bhagat | CONTENT: He completed his school years at The Army Public School, Dhaula Kuan in Delhi. He received his undergraduate degree in mechanical engineering from the Indian Institute of Technology Delhi in 1995 and his MBA degree from the Indian Institute of Management Ahmedabad in 1997. Bhagat recounted in an interview with Newslaundry that he applied after his studies to the investment banking company Goldman Sachs, where he was finally selected after 27 internal interviews.[8] | END ID: 297

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 298 | TITLE: Member of parliament | CONTENT: A member of Parliament is known as deputado, a person who is appointed after democratic election to act on people's behalf. The parliament is called Assembleia da República. | END ID: 298

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 299 | TITLE: Hard disk drive | CONTENT: Examples of partition mapping scheme include Master boot record (MBR) and GUID Partition Table (GPT). Examples of data structures stored on disk to retrieve files include the File Allocation Table (FAT) in the DOS file system and inodes in many UNIX file systems, as well as other operating system data structures (also known as metadata). As a consequence, not all the space on an HDD is available for user files, but this system overhead is usually small compared with user data. | END ID: 299

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 300 | TITLE: Baby (Justin Bieber song) | CONTENT: The song is predominantly upbeat, featuring Bieber's R&B vocals over a backdrop containing a dance infused beat, full of keyboard and "disco string" synths.[7] The song is composed in the key of E♭ major with Bieber's vocal range spanning from the low-note of G3 to the high-note of C5.[8][9] According to Jody Rosen of Rolling Stone, the song "blends winks at Fifties doo-wop with hip-hop chants", comparing the style and the lyrics "My first love broke my heart for the first time/And I was like/Baby, baby, baby, ooooh/I thought you'd always be mine" to fifties ballads like "Tears on My Pillow", "Why Do Fools Fall in Love" and "Earth Angel".[9] Lyrically, Bieber's lines explain his distress over his lost love, and promise to get it back, featured in lines like, "And I wanna play it cool/But I'm losin' you…/I'm in pieces/So come and fix me…".[7] The chorus features the distinct and repetitive "baby, baby, baby, ohhhh (nooooo)" hook. After the second verse, Ludacris comes in with the verse-rap, an anecdote of young love when he was thirteen, as it runs "When I was 13/I had my first love/She had me going crazy/Oh, I was star-struck/She woke me up daily/Don't need no Starbucks…".[10] | END ID: 300

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 301 | TITLE: Boston | CONTENT: Several historic sites relating to the American Revolution period are preserved as part of the Boston National Historical Park because of the city's prominent role. Many are found along the Freedom Trail, which is marked by a red line of bricks embedded in the ground. | END ID: 301

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 302 | TITLE: State of Emergency in India | CONTENT: e. Every state in India except two states, Chhattisgarh and Telangana has been under a state of emergency at some point of time or the other. The state of emergency is commonly known as 'President's Rule'. | END ID: 302

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 303 | TITLE: The Great War in England in 1897 | CONTENT: The Great War in England in 1897 was written by William Le Queux and published in 1894. | END ID: 303

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 304 | TITLE: James and the Giant Peach (film) | CONTENT: James and the Giant Peach is a 1996 British-American musical fantasy film directed by Henry Selick, based on the 1961 novel of the same name by Roald Dahl. It was produced by Tim Burton and Denise Di Novi, and starred Paul Terry as James. The film is a combination of live action and stop-motion animation. Co-stars Joanna Lumley and Miriam Margolyes played James's aunts in the live-action segments, and Simon Callow, Richard Dreyfuss, Susan Sarandon, Jane Leeves, David Thewlis, and Margolyes voiced his insect friends in the animation sequences. | END ID: 304

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 305 | TITLE: Horse care | CONTENT: The standard dimensions for a box stall (called a "box" in the UK, and a "stall" in the USA) vary from 10' by 12' to 14' by 14', depending on local cultural traditions, the breed of horse, gender, and any special needs. Mares with foals often are kept in double stalls.[5] Stallions, kept alone with less access to turnout, are also often given larger quarters. Ponies sometimes are kept in smaller box stalls, and warmbloods or draft horses may need larger ones. Horses kept in stables need daily exercise and may develop stable vices if they are not given work or turnout. Box stalls usually contain a layer of absorbent bedding such as straw or wood shavings and need to be cleaned daily; a horse generates approximately 15 pounds (6.8Â kg) of manure and several gallons of urine each day. There are health risks to the horse if forced to stand all day in its own waste. However, stables are built as much for the convenience of humans as horses; most healthy horses are equally, if not more, comfortable in a field or paddock with a simple three-sided shed that protects them from the elements. | END ID: 305

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 306 | TITLE: Perfect season | CONTENT: For other sports leagues for individuals, such as the PGA Tour or NASCAR, a perfect season would represent winning every event in a season. Considering the number of tournaments or races in those leagues, and the fact that each individual faces over 40 opponents as opposed to one, a perfect season is almost impossible. | END ID: 306

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 307 | TITLE: English draughts | CONTENT: The woman's championship is more recent and started in 1993, the winners have been from Ireland, Turkmenistan, and the Ukraine. | END ID: 307

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 308 | TITLE: Systole | CONTENT: Atrial fibrillation represents a common electrical malady in the heart that appears during the time interval of atrial systole (see figure at right margin). Theory suggests that an ectopic focus, usually situated within the pulmonary trunks, competes with the sinoatrial node for electrical control of the atrial chambers and thereby diminishes the performance of the atrial myocardium, or atrial heart muscle. The ordered, sinoatrial control of atrial electrical activity is disrupted, causing the loss of coordinated generation of pressure in the two atrial chambers. Atrial fibrillation represents an electrically-disordered but well perfused atrial mass working (in an uncoordinated fashion) with a (comparatively) electrically-healthy ventricular systole. | END ID: 308

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 309 | TITLE: 2016–17 Tottenham Hotspur F.C. season | CONTENT: Win   Draw   Loss | END ID: 309

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 310 | TITLE: 47 (number) | CONTENT: The Brooklyn-based hip hop collective Pro Era and its late co-founder Jamal Dewar, better known by his stage name Capital Steez, have made references to the number 47 in various songs by members of the group. The design of one of Pro Era's logos is the number 47 with its digits joined together.[20] The origins of the group's connection with the number can be linked to the production of Capital Steez's 2012 debut mixtape AmeriKKKan Korruption. The rapper was heavily fixated with the number during that time; he felt that 47 was a perfect expression of balance in the world, representing the tension between the heart and the brain (the fourth and seventh chakra, respectively.)[21] | END ID: 310

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 311 | TITLE: EDF Energy | CONTENT: EDF Energy is an integrated energy company in the United Kingdom, with operations spanning electricity generation and the sale of gas and electricity to homes and businesses throughout the United Kingdom. It employs 13,331 people and handles 5.7 million customer accounts.[1] [2] [3] | END ID: 311

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 312 | TITLE: Timeline of labor issues and events | CONTENT: 7 February 1894 (United States) | END ID: 312

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 313 | TITLE: Culture of Bhutan | CONTENT: Each individual dance takes up to several hours to complete and the entire set may last two to four days. Observation of the dances directly blesses the audience and also serves to transmit principles of Tantric Buddhism to the villagers. A number of the dances can be traced directly back to Shabdrung Ngawang Namgyal himself, the founder of Bhutan, and have been passed down essentially unchanged since the mid-17th century. | END ID: 313

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 314 | TITLE: Baron | CONTENT: The Scottish equivalent of an English baron is a Lord of Parliament.[7] | END ID: 314

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 315 | TITLE: Scheduled Banks (India) | CONTENT: Scheduled Banks in India refer to those banks which have been included in the Second Schedule of Reserve Bank of India Act, 1934. RBI in turn includes only those banks in this Schedule which satisfy the criteria laid down vide section 42(6)(a) of the said Act. Banks not under this Schedule are called Non-Scheduled Banks. | END ID: 315

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 316 | TITLE: 1960 U-2 incident | CONTENT: At the summit, after Khrushchev had blown Eisenhower's cover, Eisenhower did not deny that the plane had been spying on Soviet military installations but contended that the action was not aggressive but defensive. He argued that the current state of international relations was not one in which peaceful coexistence was an already established fact. The policy of the United States towards the Soviet Union at that time was defensive and cautionary. Eisenhower also made the point that dialogue at the Four Powers Summit was the type of international negotiation that could lead to a relationship between the United States and the Soviet Union where there would be no need to spy on each other. Eisenhower also laid out a plan for an international agreement that authorized the U.N. to "inspect" any nations willing to submit to its authority for signs of increased militarization and aggressive action. He stated that the United States would be more than willing to submit to such an inspection by the U.N. and that he hoped to circumvent the spying controversy with this alternative international surveillance agreement.[37] | END ID: 316

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 317 | TITLE: China–India relations | CONTENT: In more modern times, China and India have been working together to produce films together, such as Kung Fu Yoga starring Jackie Chan.[180] However, disruptions have risen again due to China building trade routes with Pakistan on disputed Kashmir territory.[181] | END ID: 317

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 318 | TITLE: Trolls: The Beat Goes On! | CONTENT: Trolls: The Beat Goes On! is a 2018 American animated television series produced by DreamWorks Animation that is based on the 3D computer-animated musical romantic comedy film Trolls. The series premiered on Netflix on January 19, 2018 exclusively in the United States, Canada, Latin America, United Kingdom, Ireland, Australia, New Zealand, the Nordics, Benelux, and France.[1] Amanda Leighton, Skylar Astin, Kari Wahlgren, Sam Lerner, David Kaye, David Fynn, Sean T. Krishnan, Kevin Michael Richardson, and Fryda Wolff provide the new voices for Poppy, Branch, Bridget, King Gristle, King Peppy, Biggie, Guy Diamond, Smidge, and DJ Suki and Satin & Chenille for this series respectively; only Ron Funches and Walt Dohrn reprise their roles as Cooper and Cloud Guy, also respectively.[2] Matt Lowe also voices Creek in the series, who returns in "Creek Week". | END ID: 318

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 319 | TITLE: Cost accounting | CONTENT: (20/10) | END ID: 319

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 320 | TITLE: The Gambia | CONTENT: Due to a small amount of immigrants from South Asia, Hindus and followers of the Bahá'í Faith are also present.[77] However, the vast majority of South Asian immigrants are Muslim.[77] | END ID: 320

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 321 | TITLE: Engine balance | CONTENT: Flat six engine with 180° phase offset between opposing cylinder pair, and 120° phase offset among the three pairs (these are called Boxer Six engine) is the common configuration. These 6 cylinder Boxer engines have 14. (Plane imbalance on torque generation) and 16. (Plane imbalance on compression) just like in inline six. As the strength of vibration generated by these imbalances are more or less proportional to engine length, boxer six has the advantage as flat-6 is much shorter than an inline 6 configuration. However, boxer six has additional plane imbalances on rotating mass (4.) and reciprocating mass (6.) due to its left and right banks being staggered front to back, although the offset distance tends to be much smaller in relation to the engine size than in flat-four and flat-twin. | END ID: 321

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 322 | TITLE: Jason (given name) | CONTENT: Its popularity in the United Kingdom peaked during the 1970s, when it was among the top 20 male names, but it had fallen out of the top 100 by 2003.[9] | END ID: 322

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 323 | TITLE: Walton-on-Thames railway station | CONTENT: Rush hour services to London Waterloo only operate in the morning, and services to Woking and Guildford operate in the evening rush hour with one service running semi fast to Basingstoke. | END ID: 323

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 324 | TITLE: List of military special forces units | CONTENT: Security Service of Ukraine | END ID: 324

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 325 | TITLE: Feast of the Transfiguration | CONTENT: Grapes are traditionally brought to church to be blessed after the Divine Liturgy on the day of the Transfiguration. If grapes are not available in the area, apples or some other fruit may be brought. This begins the "Blessing of First Fruits" for the year. | END ID: 325

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 326 | TITLE: Boeing 747 | CONTENT: The 747's maximum takeoff weight ranges from 735,000 pounds (333,400 kg) for the -100 to 970,000 lb (439,985 kg) for the -8. Its range has increased from 5,300 nautical miles (6,100 mi, 9,800 km) on the -100 to 8,000 nmi (9,200 mi, 14,815 km) on the -8I.[114][115] | END ID: 326

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 327 | TITLE: Untitled (How Could This Happen to Me?) | CONTENT: This is the story we wanted to tell with this video: the story of all the innocent victims caused by drinking and driving. We hope you will take the time to watch the video. Thanks for all your support. | END ID: 327

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 328 | TITLE: Abrogation doctrine | CONTENT: Another limitation that the courts have read into Congressional power to abrogate is the "congruence and proportionality" test, first discussed in City of Boerne v. Flores, 521 U.S. 507 (1997). Because the Fourteenth Amendment allows Congress to take "appropriate" action to enforce rights, the Court has determined that such action must be congruent and proportional to the deprivation of the right that the Congress is seeking to remedy. An example of a case where an act of the Congress failed the Boerne test is Kimel v. Florida Board of Regents, 528 U.S. 62 (2000). An example where an act passed the Boerne test is Nevada Department of Human Resources v. Hibbs, 538 U.S. 721 (2003). | END ID: 328

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 329 | TITLE: Stuxnet | CONTENT: Different variants of Stuxnet targeted five Iranian organizations,[20] with the probable target widely suspected to be uranium enrichment infrastructure in Iran;[19][21][22] Symantec noted in August 2010 that 60% of the infected computers worldwide were in Iran.[23] Siemens stated that the worm has not caused any damage to its customers,[24] but the Iran nuclear program, which uses embargoed Siemens equipment procured secretly, has been damaged by Stuxnet.[25][26] Kaspersky Lab concluded that the sophisticated attack could only have been conducted "with nation-state support".[27] This was further supported by the F-Secure's chief researcher Mikko Hyppönen who commented in a Stuxnet FAQ, "That's what it would look like, yes".[28] | END ID: 329

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 330 | TITLE: Fresno, California | CONTENT: The neighborhood features restaurants, live theater and nightclubs, as well as several independent shops and bookstores, currently operating on or near Olive Avenue, and all within a few hundred feet of each other. Since renewal, the Tower District has become an attractive area for restaurant and other local businesses. Today, the Tower District is also known as the center of Fresno's LGBT and hipster Communities.;[37] Additionally, Tower District is also known as the center of Fresno's local punk/goth/deathrock and heavy metal community.[citation needed] | END ID: 330

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 331 | TITLE: Kaun Banega Crorepati | CONTENT: (2000â€“01) | END ID: 331

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 332 | TITLE: History of Latvia | CONTENT: Soviet nomenklatura sanatorium in Jūrmala | END ID: 332

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 333 | TITLE: Papal infallibility | CONTENT: Following the 1869â€“1870 First Vatican Council, dissent arose among a few Catholics, almost exclusively German, Austrian, and Swiss, over the definition of papal infallibility. The dissenters, while holding the General Councils of the Church infallible, were unwilling to accept the dogma of papal infallibility, and thus a schism arose between them and the Church, resulting in the formation of communities in schism with Rome, which became known as the Old Catholic Churches. The vast majority of Catholics accepted the definition.[86] | END ID: 333

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 334 | TITLE: Antioch | CONTENT: Two routes from the Mediterranean, lying through the Orontes gorge and the Beilan Pass, converge in the plain of the Antioch Lake (BalÃ¼k Geut or El Bahr) and are met there by | END ID: 334

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 335 | TITLE: Fire services in the United Kingdom | CONTENT: British Nuclear Fuels and some other nuclear power station operators have their own on-site fire service. | END ID: 335

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 336 | TITLE: Thersites | CONTENT: "He got up in the assembly and attacked Agamemnon in the words of Achilles [calling him greedy and a coward] . . . Odysseus then stood up, delivered a sharp rebuke to Thersites, which he coupled with a threat to strip him naked, and then beat him on the back and shoulders with Agamemnon's sceptre; Thersites doubled over, a warm tear fell from his eye, and a bloody welt formed on his back; he sat down in fear, and in pain gazed helplessly as he wiped away his tear; but the rest of the assembly was distressed and laughed . . . There must be a figuration of wickedness as self-evident as Thersites-- the ugliest man who came to Troy-- who says what everyone else is thinking".[3] | END ID: 336

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 337 | TITLE: Wake in Fright (miniseries) | CONTENT: Wake in Fright is an Australian psychological thriller miniseries based on Kenneth Cook's 1961 novel of the same name, which first aired on Network Ten in October 2017. Directed by Kriv Stenders and written by Stephen M. Irwin, the series features an ensemble cast that includes Sean Keenan, Alex Dimitriades, Caren Pistorius, David Wenham, Anna Samson, Gary Sweet and Robyn Malcolm. | END ID: 337

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 338 | TITLE: Robert Louis Stevenson | CONTENT: A garden was designed by the Bournemouth Corporation in 1957 as a memorial to Stevenson, on the site of his Westbourne house, "Skerryvore", which he occupied from 1885 to 1887. A statue of the Skerryvore lighthouse is present on the site. | END ID: 338

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 339 | TITLE: Seabed | CONTENT: Seabed topography is flat where sedimentation is heavy and covers the tectonic features. Sediments comes from various sources: | END ID: 339

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 340 | TITLE: Acetylcholinesterase | CONTENT: In mammals, acetylcholinesterase is encoded by a single AChE gene while some invertebrates have multiple acetylcholinesterase genes. Note higher vertebrates also encode a closely related paralog BCHE (butyrylcholinesterase) with 50% amino acid identity to ACHE (doi: 10.1016/j.neuint.2012.06.016). Diversity in the transcribed products from the sole mammalian gene arises from alternative mRNA splicing and post-translational associations of catalytic and structural subunits. There are three known forms: T (tail), R (read through), and H(hydrophobic).[35] | END ID: 340

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 341 | TITLE: Modern Paganism | CONTENT: Eclectic Paganism takes an undogmatic religious stance,[55] and therefore potentially see no one as having authority to deem a source apocryphal. Contemporary paganism has therefore been prone to fakelore, especially in recent years as information and misinformation alike have been spread on the Internet and in print media. A number of Wiccan, pagan and even some Traditionalist or Tribalist groups have a history of Grandmother Stories – typically involving initiation by a Grandmother, Grandfather, or other elderly relative who is said to have instructed them in the secret, millennia-old traditions of their ancestors. As this secret wisdom can almost always be traced to recent sources, tellers of these stories have often later admitted they made them up.[56] Strmiska asserts that contemporary paganism could be viewed as a part of the "much larger phenomenon" of efforts to revive "traditional, indigenous, or native religions" that were occurring across the globe.[β] | END ID: 341

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 342 | TITLE: My Country, 'Tis of Thee | CONTENT: "My Country, 'Tis of Thee", also known as "America", is an American patriotic song, whose lyrics were written by Samuel Francis Smith.[2] The melody used is the same as that of the national anthem of the United Kingdom, "God Save the Queen", arranged by Thomas Arne. The song served as one of the de facto national anthems of the United States (along with songs like "Hail, Columbia") before the adoption of "The Star-Spangled Banner" as the official U.S. national anthem in 1931.[3] | END ID: 342

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 343 | TITLE: Dominican Republic | CONTENT: Haiti's constitution forbade white elites from owning land, and Dominican major landowning families were forcibly deprived of their properties. Many emigrated to Cuba, Puerto Rico (these two being Spanish possessions at the time), or Gran Colombia, usually with the encouragement of Haitian officials who acquired their lands. The Haitians associated the Roman Catholic Church with the French slave-masters who had exploited them before independence and confiscated all church property, deported all foreign clergy, and severed the ties of the remaining clergy to the Vatican. | END ID: 343

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 344 | TITLE: Hayling Island | CONTENT: The natural beach at Hayling was predominantly sandy, but in recent years it has been mechanically topped with shingle dredged from the bed of the Solent in an effort to reduce beach erosion and reduce the potential to flood low-lying land. At low tide, the East Winner sandbank is visible, extending a mile out to sea. The coastline in this area has substantially changed since Roman times: it is believed much land has been lost from the coasts of Hayling and Selsey by erosion and subsequent flooding. | END ID: 344

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 345 | TITLE: Big Ten Conference Women's Basketball Player of the Year | CONTENT: The Big Ten Conference Women's Basketball Player of the Year is a basketball award given to the Big Ten Conference's most outstanding player. The award was first given following the Big Ten's first full season of women's basketball in 1982â€“83 (although the conference held its first postseason tournament the previous season). The league's head coaches have presented the award since 1983; media members who cover Big Ten women's basketball began presenting their own version of the award in 1996.[n 1] | END ID: 345

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 346 | TITLE: Brevard Public Schools | CONTENT: After the collapse of property values, the district sought to save money by deferring maintenance. For example, instead of replacing each of its 409 buses every 12 years, it went to a 55-year replacement cycle.[35] | END ID: 346

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 347 | TITLE: The Strain (TV series) | CONTENT: The pilot episode began principal photography on September 17, 2013, in Toronto, Ontario, Canada.[52][53] Shooting of the pilot was finished on October 31, 2013. FX ordered 13 episodes. Season one was expected to film from November 25, 2013, to April 30, 2014.[54] A full writing staff was hired to script subsequent episodes. FX reportedly committed $500,000 to creature creation.[11] Twelve swords used in the series were provided by Missoula, Montana-based bladesmith company Zombie Tools.[55] Production began for the second season in Toronto in November 2014.[56] | END ID: 347

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 348 | TITLE: Carnatic music | CONTENT: Kalpanaswaras have a somewhat predictable rhythmical structure;[42] the swaras are sung to end on the samam (the first beat of the rhythmical cycle).[38] The swaras can also be sung at the same speed or double the speed of the melody that is being sung, though some artists sing triple-speed phrases too.[38] | END ID: 348

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 349 | TITLE: Watching Movies with the Sound Off | CONTENT: On June 25, 2013, the music video was released for "Objects in the Mirror".[37] Then on July 8, 2013, the music video was released for "Gees" featuring Schoolboy Q.[38] Two weeks later, the music video was released for "I Am Who Am (Killin Time)" featuring Niki Randa.[39] "The Star Room" received video treatment, being premiered on October 2, 2013.[40] The video for "Youforia" followed later that month.[41] Then on February 15, 2014, the music video was released for "Avian".[42] | END ID: 349

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 350 | TITLE: Military dictatorship of Chile (1973–90) | CONTENT: Financial conglomerates became major beneficiaries of the liberalized economy and the flood of foreign bank loans. Large foreign banks reinstated the credit cycle, as the Junta saw that the basic state obligations, such as resuming payment of principal and interest installments, were honored. International lending organizations such as the World Bank, the International Monetary Fund, and the Inter-American Development Bank lent vast sums anew.[78] Many foreign multinational corporations such as International Telephone and Telegraph (ITT), Dow Chemical, and Firestone, all expropriated by Allende, returned to Chile.[78] | END ID: 350

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 351 | TITLE: The Tale of Despereaux | CONTENT: A noble mouse named Despereaux saves a princess. | END ID: 351

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 352 | TITLE: Propranolol | CONTENT: Propranolol is classified as a non-cardioselective sympatholytic beta blocker that crosses the blood–brain barrier. It is lipid soluble and also has sodium channel blocking effects. Propranolol is a non-selective beta blocker; that is, it blocks the action of epinephrine (adrenaline) and norepinephrine (noradrenaline) at both β1- and β2-adrenergic receptors. It has little intrinsic sympathomimetic activity, but has strong membrane stabilizing activity (only at high blood concentrations, e.g. overdose).[citation needed] Propranolol is able to cross the blood–brain barrier and exert effects in the central nervous system in addition to its peripheral activity.[21] | END ID: 352

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 353 | TITLE: Adaptation | CONTENT: In natural theology, adaptation was interpreted as the work of a deity and as evidence for the existence of God.[1] William Paley believed that organisms were perfectly adapted to the lives they led, an argument that shadowed Gottfried Wilhelm Leibniz, who had argued that God had brought about "the best of all possible worlds." Voltaire's Dr. Pangloss[2] is a parody of this optimistic idea, and David Hume also argued against design.[3] The Bridgewater Treatises are a product of natural theology, though some of the authors managed to present their work in a fairly neutral manner. The series was lampooned by Robert Knox, who held quasi-evolutionary views, as the Bilgewater Treatises. Charles Darwin broke with the tradition by emphasising the flaws and limitations which occurred in the animal and plant worlds.[4] | END ID: 353

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 354 | TITLE: Ministry of Corporate Affairs | CONTENT: The Ministry of Corporate Affairs (MCA) is an Indian government ministry.This Ministry is primarily concerned with administration of the Companies Act 2013, the Companies Act 1956, the Limited Liability Partnership Act, 2008 & other allied Acts and rules & regulations framed there-under mainly for regulating the functioning of the corporate sector in accordance with law.[1] It is responsible mainly for regulation of Indian enterprises in Industrial and Services sector. The current minister of corporate affairs is Arun Jaitley. The current Minister of State for Corporate Affairs is Mr. PP Choudhary. | END ID: 354

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 355 | TITLE: Fayolism | CONTENT: According to Claude George (1968), a primary difference between Fayol and Taylor was that Taylor viewed management processes from the bottom up, while Fayol viewed it from the top down. In Fayol's book General and Industrial Management, Fayol wrote that | END ID: 355

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 356 | TITLE: Diocese | CONTENT: In the United Methodist Church (the United States and some other countries), a bishop is given oversight over a geographical area called an episcopal area. Each episcopal area contains one or more annual conferences, which is how the churches and clergy under the bishop's supervision are organized. Thus, the use of the term "diocese" referring to geography is the most equivalent in the United Methodist Church, whereas each annual conference is part of one episcopal area (though that area may contain more than one conference). The African Methodist Episcopal Church has a similar structure to the United Methodist Church, also using the Episcopal Area. Note that the bishops govern the church as a single bench.[citation needed] | END ID: 356

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 357 | TITLE: Sheffield Wednesday F.C. | CONTENT: Sheffield Wednesday Football Club is a professional association football club based in Sheffield, England. The team competes in the Championship, the second tier of the English football league system. Formed as an offshoot of The Wednesday Cricket Club in 1867, they went by the name of The Wednesday Football Club until changing to their current name in 1929. | END ID: 357

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 358 | TITLE: History of the United Arab Emirates | CONTENT: In the 1930s, the first oil company teams carried out preliminary surveys. An onshore concession was granted to Petroleum Development (Trucial Coast) in 1939, and an offshore concession to D'Arcy Exploration Ltd in 1952.[11] Oil was discovered under an old pearling bed in the Persian Gulf, Umm Shaif, in 1958, and in the desert at Murban in 1960. The first cargo of crude was exported from Jabel Dhanna in Abu Dhabi in 1962. As oil revenues increased, the ruler of Abu Dhabi, Sheikh Zayed bin Sultan Al Nahyan, undertook a massive construction program, building schools, housing, hospitals and roads. When Dubai's oil exports commenced in 1969, Sheikh Rashid bin Saeed Al Maktoum, the ruler of Dubai, was also able to use oil revenues to improve his people's quality of life.[12] | END ID: 358

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 359 | TITLE: Ascending aorta | CONTENT: Porcelain aorta is extensive atherosclerotic calcification of the ascending aorta.[6] It makes aortic surgery difficult, especially aortic cross-clamping, and incisions may result in excessive aortic injury and/or arterial embolism.[6] | END ID: 359

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 360 | TITLE: Crowne Plaza | CONTENT: Crowne Plaza is a multinational chain of full service, upscale hotels catering to business travelers and to the meetings and conventions market. It forms part of the InterContinental Hotels Group family of brands, which include InterContinental Hotels & Resorts and Holiday Inn Hotels & Resorts, and operates in 52 countries with more than 400 hotels, usually located in city centers, resorts, coastal towns or near major airports. | END ID: 360

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 361 | TITLE: Voice over LTE | CONTENT: In September 2017, Swazi Mobile was the first to activate VoLTE in Swaziland. | END ID: 361

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 362 | TITLE: Alcoholic drink | CONTENT: Alcoholic drinks are a source of food energy. The USDA uses a figure of 6.93 kilocalories (29.0 kJ) per gram of alcohol (5.47 kcal (22.9 kJ) per ml) for calculating food energy.[27] In addition to alcohol, many alcoholic drinks contain carbohydrates. For example, along with approximately 96 calories from alcohol in 12 US fl oz (355 ml) of 5% ABV beer, there are usually 10–15 g of carbohydrates (40–60 kcal or 170–250 kJ).[citation needed] Excessive daily calorie intake may contribute to an increase in body weight and so-called "beer belly". In addition to the direct effect of its caloric content, alcohol is also known to potentiate the insulin response of the human body to glucose, which, in essence, "instructs" the body to convert consumed carbohydrates into fat and to suppress carbohydrate and fat oxidation.[28][29] Ethanol is directly processed in the liver to acetyl CoA, the same intermediate product as in glucose metabolism. Because ethanol can only be metabolized and consumed by the liver, chronic excessive use can lead to fatty liver. This leads to a chronic inflammation of the liver and eventually alcoholic liver disease. | END ID: 362

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 363 | TITLE: Hard water | CONTENT: Ryznar saturation index (RSI) was developed from empirical observations of corrosion rates and film formation in steel mains. It is defined as: | END ID: 363

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 364 | TITLE: Glossary of comics terminology | CONTENT: "Underground comix" is a term first popularized by cartoonists in the underground comix movement of the 1960s and 1970s in an attempt to move the word away from its etymological origins.  Art Spiegelman in particular has been a proponent of its usage, hoping to highlight the fact that the medium is capable of mature, non-comedic content, as well as to emphasize the hybrid nature of the medium ("co-mix").[3] | END ID: 364

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 365 | TITLE: File format | CONTENT: Hiding the extension, however, can create the appearance of two or more identical filenames in the same folder. For example, a company logo may be needed both in .eps format (for publishing) and .png format (for web sites). With the extensions visible, these would appear as the unique filenames "CompanyLogo.eps" and "CompanyLogo.png". On the other hand, hiding the extensions would make both appear as "CompanyLogo". | END ID: 365

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 366 | TITLE: Between a Rock and a Hard Place | CONTENT: Between a Rock and a Hard Place may refer to: | END ID: 366

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 367 | TITLE: 2017 New Year Honours | CONTENT: Royal Air Force | END ID: 367

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 368 | TITLE: Far Beyond the Stars | CONTENT: "Far Beyond the Stars" is the 137th episode of the syndicated science fiction television series Star Trek: Deep Space Nine, the 13th episode of season six. The teleplay was written by Ira Steven Behr and Hans Beimler, based on a story by Marc Scott Zicree. Series star Avery Brooks directed. It is unique in that almost the full cast of DS9 portrays human characters, without their alien costumes, as a rare example of metafiction in the fictional Star Trek universe. | END ID: 368

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 369 | TITLE: Population decline | CONTENT: The economies of both Japan and Germany both went into recovery around the time their populations just began to decline (2003–2006). In other words, both the total and per capita GDP in both countries grew more rapidly after 2005 than before. Russia's economy also began to grow rapidly from 1999 onward, even though its population has been shrinking since 1992-93 (the decline is now decelerating).[46] In addition, many Eastern European countries have been experiencing similar effects to Russia. Such renewed growth calls into question the conventional wisdom that economic growth requires population growth, or that economic growth is impossible during a population decline. However, it may be argued that this renewed growth is in spite of population decline rather than because of it, and economic growth in these countries would potentially be greater if they were not undergoing such demographic decline. For example, Russia has become quite wealthy selling fossil fuels such as oil, which are now high-priced, and in addition, its economy has expanded from a very low nadir due to the economic crisis of the late 1990s. And although Japan and Germany have recovered somewhat from having been in a deflationary recession and stagnation, respectively, for the past decade, their recoveries seem to have been quite tepid. Both countries fell into the global recession of 2008–2009, but are now recovering once again, being among the first countries to recover.[47][48] | END ID: 369

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 370 | TITLE: Finding Dory | CONTENT: It is now the highest-grossing Disney animated or Pixar film in Australia (where it is also the second highest-grossing animated film of all time behind Shrek 2), Bolivia, Brazil, Central America, Colombia, India, Indonesia, New Zealand, Peru, the Philippines, and Trinidad.[98][100][112] It also became the second highest-grossing Pixar release of all time in South Korea behind Inside Out.[113] Elsewhere, the biggest markets in terms of total earnings were Japan ($66 million), followed by the UK ($56.3 million), China ($38.1 million), Australia ($36.3 million), and Brazil ($34.5 million).[48] | END ID: 370

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 371 | TITLE: Photoperiodism | CONTENT: Some long-day facultative plants are: | END ID: 371

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 372 | TITLE: Piano pedals | CONTENT: The only piano Mozart ever owned was one by Anton Walter, c. 1782-1785. It had two knee levers; the one on the left raised all the dampers, while the one on the right raised only the treble dampers. A moderator stop to produce a softer sound (see Other pedals, above) was centrally above the keyboard.[25] | END ID: 372

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 373 | TITLE: G. Callen | CONTENT: G. Callen (born: Grisha Alekandrovich Nikolaev) is a fictional character in the show NCIS: Los Angeles portrayed by Chris O'Donnell. He is an NCIS Special Agent in Charge, and the senior agent assigned to the Office of Special Projects. O'Donnell made his first appearance during NCIS' sixth season episode "Legend (Part 1)". | END ID: 373

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 374 | TITLE: Disulfide | CONTENT: In eukaryotic cells, in general, stable disulfide bonds are formed in the lumen of the RER (rough endoplasmic reticulum) and the mitochondrial intermembrane space but not in the cytosol. This is due to the more oxidizing environment of the aforementioned compartments and more reducing environment of the cytosol (see glutathione). Thus disulfide bonds are mostly found in secretory proteins, lysosomal proteins, and the exoplasmic domains of membrane proteins. | END ID: 374

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 375 | TITLE: Reign of Terror | CONTENT: On 24 June, the convention adopted the first republican constitution of France, the French Constitution of 1793. It was ratified by public referendum, but never put into force. | END ID: 375

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 376 | TITLE: Andhra Pradesh | CONTENT: Archaeological evidence from places such as Amaravati, Dharanikota and Vaddamanu suggests that the Andhra region was part of the Mauryan Empire. Amaravati might have been a regional centre for Mauryan rule. After the death of Emperor Ashoka, Mauryan rule weakened around 200 BCE, and was replaced by several smaller kingdoms in the Andhra region.[23] | END ID: 376

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 377 | TITLE: Tom Felton | CONTENT: Felton had a cameo role in Get Him to the Greek, released on 4 June 2010.[22] He portrayed the human character Dodge Landon in the 2011 science-fiction film Rise of the Planet of the Apes,[3] and played a paranormal investigator in the thriller film The Apparition (2012).[23] | END ID: 377

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 378 | TITLE: Politics of Georgia (country) | CONTENT: The Speaker of Parliament is Irakli Kobakhidze. | END ID: 378

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 379 | TITLE: Chola Navy | CONTENT: Towards the end of the 9th century, southern India had developed extensive maritime and commercial activity, especially with the Chinese and Arabs.[16][18] The Cholas, being in possession of parts of both the west and the east coasts of peninsular India, were at the forefront of these ventures.[19][20][21] The Tang dynasty of China, the Srivijaya empire in the Malayan archipelago under the Sailendras, and the Abbasid Kalifat at Baghdad were the main trading partners.[22] | END ID: 379

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 380 | TITLE: Monarchy of New Zealand | CONTENT: There are a number of legal issues to be addressed in order to abolish the monarchy,[102] though individuals on both sides of the argument take a different view of the level of difficulty faced.[103] Much of the unsurety involves the reserve powers of the sovereign; the relationship between the various regions of the Realm of New Zealand presently sharing the same sovereign (the absence of these matters from republican arguments having been criticised as a "self-centredness of republican discussions in New Zealand"[64] | END ID: 380

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 381 | TITLE: The Neutral Zone (Star Trek: The Next Generation) | CONTENT: While Captain Picard (Patrick Stewart) is away at an emergency Federation conference, the Enterprise crew discovers an ancient space capsule from Earth. Inside they find three humans in cryonic chambers. Lt. Cdr. Data (Brent Spiner) asks to move the chambers to the Enterprise and Commander Riker (Jonathan Frakes) agrees. Picard returns and orders the Enterprise to the Neutral Zone, as several Federation outposts near the edges of the zone have not responded to communications. He explains that the conference was about the potential threat of the Romulans, who have not been seen for the last several decades. As Data and Chief Medical Officer Dr. Crusher (Gates McFadden) work to thaw the cryonically preserved humans, Picard admonishes Data for bringing them aboard during a crucial time, and puts Riker in charge of looking after them. | END ID: 381

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 382 | TITLE: Napoleon | CONTENT: JosÃ©phine had lovers, such as Lieutenant Hippolyte Charles, during Napoleon's Italian campaign.[302] Napoleon learnt of that affair and a letter he wrote about it was intercepted by the British and published widely, to embarrass Napoleon. Napoleon had his own affairs too: during the Egyptian campaign he took Pauline Bellisle Foures, the wife of a junior officer, as his mistress. She became known as "Cleopatra".[303][note 9] | END ID: 382

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 383 | TITLE: Oscar De La Hoya | CONTENT: In September, 2007, De La Hoya's company Golden Boy Enterprises acquired The Ring, KO Magazine, and World Boxing Magazine from Kappa Publishing Group.[52] | END ID: 383

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 384 | TITLE: You'll Never Walk Alone | CONTENT: "You'll Never Walk Alone" is a show tune from the 1945 Rodgers and Hammerstein musical Carousel. In the second act of the musical, Nettie Fowler, the cousin of the female protagonist Julie Jordan, sings "You'll Never Walk Alone" to comfort and encourage Julie when her husband, Billy Bigelow, the male lead, commits suicide after a failed robbery attempt. It is reprised in the final scene to encourage a graduation class of which Louise (Billy and Julie's daughter) is a member. The now invisible Billy, who has been granted the chance to return to Earth for one day in order to redeem himself, watches the ceremony and is able to silently motivate the unhappy Louise to join in the song. | END ID: 384

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 385 | TITLE: Right of foreigners to vote | CONTENT: In 1960, non-citizen voting rights in local elections were granted for holders of a permanent resident card ("blue card").[2] Most permanent residents, a status created by the 1952 Entry into Israel Law, are migrants, but other groups fall into the same category. | END ID: 385

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 386 | TITLE: Mexico at the 2018 Winter Olympics | CONTENT: Mexico competed at the 2018 Winter Olympics in Pyeongchang, South Korea, from 9 to 25 February 2018, with four competitors in three sports. | END ID: 386

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 387 | TITLE: Hawaiian sovereignty movement | CONTENT: David Keanu Sai and Kamana Beamer are two Hawaiian scholars whose works use international law to argue for the rights of a Hawaiian Kingdom existing today and call for an end to US occupation of the islands.[45]:394 Trained as a U.S. military officer, Sai uses the title of Chairman of the Acting Council of Regency of the Hawaiian Kingdom organization.[77] Sai has done extensive historical research, especially on the treaties between Hawaii and other nations, and on military occupation and the laws of war. Dr. Keanu Sai teaches Hawaiian Studies at Windward Community College.[78] | END ID: 387

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 388 | TITLE: List of career achievements by Kareem Abdul-Jabbar | CONTENT: This is page of achievements of greatest basketball player of all time - Kareem Abdul-Jabbar. | END ID: 388

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 389 | TITLE: Pro Football Hall of Fame | CONTENT: With the election of the Class of 2018[2] – Bobby Beathard, Robert Brazile, Brian Dawkins, Jerry Kramer, Ray Lewis, Randy Moss, Terrell Owens, and Brian Urlacher – there are a total of 318 members of the Hall of Fame.[3] | END ID: 389

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 390 | TITLE: Wok | CONTENT: A wok cleaning brush made of bamboo. | END ID: 390

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 391 | TITLE: Protein biosynthesis | CONTENT: Transcription occurs in the cell nucleus, where the DNA is held. The DNA structure of the cell is made up of two helixes made up of sugar and phosphate held together by hydrogen bonds between the bases of opposite strands. The sugar and the phosphate in each strand are joined together by stronger phosphodiester covalent bonds. The DNA is "unzipped" (disruption of hydrogen bonds between different single strands) by the enzyme helicase, leaving the single nucleotide chain open to be copied. RNA polymerase reads the DNA strand from the 3-prime (3') end to the 5-prime (5') end, while it synthesizes a single strand of messenger RNA in the 5'-to-3' direction. The general RNA structure is very similar to the DNA structure, but in RNA the nucleotide uracil takes the place that thymine occupies in DNA. The single strand of mRNA leaves the nucleus through nuclear pores, and migrates into the cytoplasm. | END ID: 391

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 392 | TITLE: Samarium | CONTENT: Another samarium-based laser became the first saturated X-ray laser operating at wavelengths shorter than 10 nanometers. It provided 50-picosecond pulses at 7.3 and 6.8 nm suitable for applications in holography, high-resolution microscopy of biological specimens, deflectometry, interferometry, and radiography of dense plasmas related to confinement fusion and astrophysics. Saturated operation meant that the maximum possible power was extracted from the lasing medium, resulting in the high peak energy of 0.3 mJ. The active medium was samarium plasma produced by irradiating samarium-coated glass with a pulsed infrared Nd-glass laser (wavelength ~1.05 µm).[87] | END ID: 392

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 393 | TITLE: Colonel (United States) | CONTENT: In the modern armed forces, the colonel's eagle is worn facing forward with head and beak pointing towards the wearer's front. Of all U.S. military commissioned officer rank, only the colonel's eagle has a distinct right and left insignia. All other commissioned officer rank insignia can be worn on either the right or left side. | END ID: 393

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 394 | TITLE: ISO 14000 | CONTENT: During the "check" stage, performance is monitored and periodically measured to ensure that the organization's environmental targets and objectives are being met (Martin 1998). In addition, internal audits are conducted at planned intervals to ascertain whether the EMS meets the user's expectations and whether the processes and procedures are being adequately maintained and monitored (Standards Australia/Standards New Zealand 2004). | END ID: 394

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 395 | TITLE: Vertical bar | CONTENT: The same "pipe" feature is also found in later versions of DOS and Microsoft Windows. | END ID: 395

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 396 | TITLE: Timeline of religion | CONTENT: 1939â€“1945 | END ID: 396

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 397 | TITLE: SSX (series) | CONTENT: On October 20, 2003, SSX 3 was released. It was released on all the same platforms that SSX Tricky was released on, as well as the Gizmondo, and was developed by EA Canada. | END ID: 397

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 398 | TITLE: William III of England | CONTENT: The House of Commons, with a Whig majority, quickly resolved that the throne was vacant, and that it was safer if the ruler were Protestant. There were more Tories in the House of Lords, which would not initially agree, but after William refused to be a regent or to agree to remain king only in his wife's lifetime, there were negotiations between the two houses and the Lords agreed by a narrow majority that the throne was vacant. The Commons made William accept a Bill of Rights,[78] and, on 13 February 1689, Parliament passed the Declaration of Right, in which it deemed that James, by attempting to flee, had abdicated the government of the realm, thereby leaving the throne vacant.[84] | END ID: 398

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 399 | TITLE: Water Resistant mark | CONTENT: In practice, the survivability of the watch will depend not only on the water depth, but also on the age of the sealing material, past damage, temperature, and additional mechanical stresses. | END ID: 399

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 400 | TITLE: Prussian Army | CONTENT: The Prussian Army crushed Danish forces in the Battle of Dybbøl during the Second Schleswig War (1864), allowing Prussia and Austria to claim Schleswig and Holstein, respectively. Disputes orchestrated by the Prussian Minister President, Otto von Bismarck, led to the Austro-Prussian War (1866). The needle guns of the Prussian infantry were highly successful against the Austrians, who were defeated at Königgrätz. Under the leadership of Moltke, the Prussian Army then proved victorious over France in the Franco-Prussian War (1870). Unlike the Austrians, the French had the powerful Chassepot rifle, which outclassed the Prussian needle gun. However, the Prussian artillery was effective against the French, who were frequently flanked or surrounded by the mobile Prussians. Patriotism in Prussia from the victories began to undermine liberal resistance to absolutism.[70] | END ID: 400

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 401 | TITLE: Alternative versions of Spider-Man | CONTENT: Peter Parquagh is a counterpart to Peter in the miniseries Marvel 1602, albeit without powers. In the series he acts as an apprentice to the royal spymaster Sir Nicholas Fury. A running gag involves Peter repeatedly almost getting bitten by unusual spiders, something that finally occurs at the very end. In the sequel, 1602: New World, he takes the identity of the Spider. Later, Peter's dual identity is revealed, and with the death of his beloved Virginia Dare at the hands of Norman Osborne, he returns to Europe and falls in love with Marion Jane Watson and joins her family of theater performers. During a battle with Baron Octavius, Norman Osborn, and Curtis Connors in Venice, a bystander picks up some of Peter's webbing which eventually served as the basis for the Super Soldier Serum and created Captain America in World War II in this universe.[26] While in the Globe theatre, he is attacked and killed by the super villain Morlun.[27][28] | END ID: 401

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 402 | TITLE: Ammonium iron(II) sulfate | CONTENT: It is used in the Fricke's dosemeter to measure high doses of gamma rays.[4] | END ID: 402

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 403 | TITLE: List of Kung Fu Panda characters | CONTENT: In "The Hunger Game," Ju-Long collaborated with Madame Zhou in a plot to steal food from the Valley of Peace and sell it to different markets. Visiting Madame Zhou where he ended up discovering that Madame Zhou was a silent partner to the Lao Shu. She then traps Po and Lao Shu in an underground room while she works to sell the stolen food to other markets. Po and Lao Shu got out where Madame Zhou wielded an iron whip in battle (she considers herself as the "Master of the Iron Whip"). After the explosions from Madame Zhou's bombs released all the stolen food onto the Valley of Peace, Po then used the Iron Whip to wrap up Madame Zhou and Ju-Long where he left them for the authorities. | END ID: 403

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 404 | TITLE: Sprouting | CONTENT: Sprouts are said to be rich in digestible energy, bioavailable vitamins, minerals, amino acids, proteins, and phytochemicals, as these are necessary for a germinating plant to grow.[3][4][5][6] These nutrients are essential for human health. The nutritional changes upon germination and sprouting are summarised below. | END ID: 404

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 405 | TITLE: Gun legislation in Germany | CONTENT: In Germany the possession of any firearm with a muzzle energy exceeding 7.5 Joule (~5.5 ft·lbf; for comparison, a .22LR cartridge has a muzzle energy of 159 J) requires a valid firearms ownership license for any particular weapon. The current Federal Weapons Act adopts a two-tiered approach to firearms licensing. | END ID: 405

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 406 | TITLE: Manuel A. Roxas High School | CONTENT: They are provided with elective subjects to fulfill the aim of the program; that is to give the students good grounding in Science, as well as in other subject areas. | END ID: 406

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 407 | TITLE: List of The Lion King characters | CONTENT: Nala (voiced by Moira Kelly in The Lion King, The Lion King ll: Simba's Pride, and The Lion King 1½, and Gabrielle Union in The Lion Guard) is the daughter of Sarafina, the best friend and later wife of Simba and Kiara and Kion's mother. Although she is a prominent character in The Lion King, she makes minor appearances in Simba's Pride, The Lion King 1½, and The Lion Guard. | END ID: 407

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 408 | TITLE: Standing rib roast | CONTENT: A standing rib roast, also known as prime rib, is a cut of beef from the primal rib, one of the nine primal cuts of beef. While the entire rib section comprises ribs six through 12, a standing rib roast may contain anywhere from two to seven ribs. | END ID: 408

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 409 | TITLE: GNU GRUB | CONTENT: GNU GRUB was developed from a package called the Grand Unified Bootloader (a play on Grand Unified Theory[5]). It is predominantly used for Unix-like systems. The GNU operating system uses GNU GRUB as its boot loader, as do most Linux distributions and the Solaris operating system on x86 systems, starting with the Solaris 10 1/06 release. | END ID: 409

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 410 | TITLE: West Shore School District | CONTENT: Pennsylvania public school districts budget and expend funds according to procedures mandated by the General Assembly and the Pennsylvania Department of Education (PDE). An annual operating budget is prepared by school district administrative officials. A uniform form is furnished by the PDE and submitted to the board of school directors for approval prior to the beginning of each fiscal year on July 1. | END ID: 410

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 411 | TITLE: Aaron | CONTENT: To emphasize the validity of the Levites' claim to the offerings and tithes of the Israelites, Moses collected a rod from the leaders of each tribe in Israel and laid the twelve rods over night in the tent of meeting. The next morning, Aaron's rod was found to have budded and blossomed and produced ripe almonds (Numbers 17:8).[49][50] The following chapter then details the distinction between Aaron's family and the rest of the Levites: while all the Levites (and only Levites) were devoted to the care of the sanctuary, charge of its interior and the altar was committed to the Aaronites alone (Numbers 18:1-7).[51] | END ID: 411

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 412 | TITLE: Charlotte, North Carolina | CONTENT: Some groups still pan for gold occasionally in local streams and creeks. The Reed Gold Mine operated until 1912. The Charlotte Mint was active until 1861, when Confederate forces seized it at the outbreak of the Civil War. The mint was not reopened at the war's end, but the building, albeit in a different location, now houses the Mint Museum of Art. | END ID: 412

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 413 | TITLE: Alpha Centauri | CONTENT: Alpha Centauri (α Centauri, abbreviated Alpha Cen, α Cen) is the closest star system to the Solar System, being 4.37 light-years (1.34 pc) from the Sun. It consists of three stars: Alpha Centauri A (also named Rigil Kentaurus[13]) and Alpha Centauri B, which form the binary star Alpha Centauri AB, and a small and faint red dwarf, Alpha Centauri C (also named Proxima Centauri[13]), which is loosely gravitationally bound and orbiting the other two at a current distance of about 13,000 astronomical units (0.21 ly). To the unaided eye, the two main components appear as a single point of light with an apparent visual magnitude of −0.27, forming the brightest star in the southern constellation of Centaurus and is the third-brightest star in the night sky, outshone only by Sirius and Canopus. | END ID: 413

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 414 | TITLE: Co-Cathedral of the Sacred Heart (Houston) | CONTENT: Under Father Troy Gately, in December 2006, the Co-Cathedral parish purchased the former Federal Reserve Bank Building, adjacent to the new Co-Cathedral for $5,000,000, and named it Cathedral Centre.  It will replace the 1922 Sacred Heart School building to house classrooms, offices, parish hall, youth rooms, child care center, music rooms, library, and a cafeteria. The parish is expected to spend another $2,000,000 on renovations for the new Cathedral Centre.[10] | END ID: 414

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 415 | TITLE: Jazz | CONTENT: As Davis recalls: | END ID: 415

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 416 | TITLE: Belling the Cat | CONTENT: One of the earliest versions of the story appears as a parable critical of the clergy in Odo of Cheriton's Parabolae.[5] Written around 1200, it was afterwards translated into Welsh, French and Spanish. Some time later the story is found in the work now referred to as Ysopet-Avionnet, which is largely made up of Latin poems by the 12th century Walter of England, followed by a French version dating from as much as two centuries later. It also includes four poems not found in Walter's Esopus; among them is the tale of "The Council of the Mice" (De muribus consilium facientibus contra catum). The author concludes with the scornful comment that laws are of no effect without the means of adequately enforcing them and that such parliamentary assemblies as he describes are like the proverbial mountain in labour that gives birth to a mouse.[6] | END ID: 416

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 417 | TITLE: Children in the military | CONTENT: During the armed conflict in Eastern Ukraine in 2014 Justice for Peace at Donbas documented 41 verified individual cases of child recruitment into armed formations.[174] Of those 37 concerned the participation of children in armed formations on territory not controlled by Ukraine and 4 on territory controlled by Ukraine. There were 31 further reports of child recruitment which could not be verified. Of the 37 verified cases on territory not controlled by Ukraine, 33 were boys and 4 were girls; 57% were aged 16â€“17, 35% were under 15, and age could not be determined in 8% of cases.[174] | END ID: 417

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 418 | TITLE: Star Wars: The Force Awakens | CONTENT: Daisy Ridley and John Boyega each received several nominations and accolades for their performances. They were nominated as Best Newcomers at various critics circle and associations, including the Alliance of Women Film Journalists,[396] and the Florida Film Critics Circle,[397] The Force Awakens received eleven nominations at the MTV Movie Awards, the most for the ceremony, including Movie of the Year, Best Female Performance for Ridley, Best Breakthrough Performance for Boyega, and Best Virtual Performance for Lupita Nyong'o and Andy Serkis.[398] | END ID: 418

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 419 | TITLE: Interstellar medium | CONTENT: The previous process excites more and more atoms because a de-excitation obeys Einsteinâ€™s law of coherent interactions: Variation dI of radiance I of a light beam along a path dx is dI=BIdx, where B is Einstein amplification coefficient which depends on medium. I is the modulus of Poynting vector of field, absorption occurs for an opposed vector, which corresponds to a change of sign of B. Factor I in this formula shows that intense rays are more amplified than weak ones (competition of modes). Emission of a flare requires a sufficient radiance I provided by random zero point field. After emission of a flare, weak B increases by pumping while I remains close to zero: De-excitation by a coherent emission involves stochastic parameters of zero point field, as observed close to quasars (and in polar auroras). | END ID: 419

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 420 | TITLE: Frankfurt | CONTENT: Within Frankfurt's urban area are several important companies. | END ID: 420

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 421 | TITLE: Charles M. Schwab | CONTENT: A bust-length portrait of Schwab painted in 1903 by Swiss-born American artist Adolfo Müller-Ury (1862–1947) was formerly in the Jessica Dragonette Collection at the American Heritage Center at the University of Wyoming at Laramie, but has been donated to the National Portrait Gallery in Washington, D.C. Müller-Ury also painted his nephew and namesake Charles M. Schwab (son of his brother Joseph) as a boy in a sailor suit around the same date.[14] | END ID: 421

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 422 | TITLE: Phosphorus pentoxide | CONTENT: Phosphorus pentoxide is a chemical compound with molecular formula P4O10 (with its common name derived from its empirical formula, P2O5). This white crystalline solid is the anhydride of phosphoric acid. It is a powerful desiccant and dehydrating agent. | END ID: 422

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 423 | TITLE: Saxophone | CONTENT: The saxophone family was invented by the Belgian instrument maker Adolphe Sax in 1840.[2][3] Adolphe Sax wanted to create a group or series of instruments that would be the most powerful and vocal of the woodwinds, and the most adaptive of the brass instruments, that would fill the vacant middle ground between the two sections. Sax patented the saxophone on June 28, 1846, in two groups of seven instruments each. Each series consisted of instruments of various sizes in alternating transposition. The series pitched in B♭ and E♭, designed for military bands, have proved popular and most saxophones encountered today are from this series. Instruments from the so-called "orchestral" series, pitched in C and F, never gained a foothold, and the B♭ and E♭ instruments have now replaced the C and F instruments when the saxophone is used in an orchestra. | END ID: 423

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 424 | TITLE: Native Americans in the United States | CONTENT: Their fellow soldiers often held them in high esteem, in part since the legend of the tough Native American warrior had become a part of the fabric of American historical legend. White servicemen sometimes showed a lighthearted respect toward Native American comrades by calling them "chief". The resulting increase in contact with the world outside of the reservation system brought profound changes to Native American culture. "The war", said the U.S. Indian Commissioner in 1945, "caused the greatest disruption of Native life since the beginning of the reservation era", affecting the habits, views, and economic well-being of tribal members.[112] The most significant of these changes was the opportunity—as a result of wartime labor shortages—to find well-paying work in cities, and many people relocated to urban areas, particularly on the West Coast with the buildup of the defense industry. | END ID: 424

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 425 | TITLE: History of Taiwan | CONTENT: Some British and Americans advocated the annexation of Taiwan.[41][42] In 1841 during the First Opium War in the Battle of Keelung (1841-1842) the British attempted to attack in failed efforts three times against Keelung on the northeast coast of Taiwan under Qing rule.[43][44][45][46] The ventures to seize Da'an and Keelung by the British failed.[47] The successful defense was directed by Yao Ying who led the Chinese naval forces on Taiwan.[48] On Taiwan some British were taken as prisoners by the taotai Yao Ying and interrogated for information on the west.[49] Indian and European crew members of the Nerbudda, a British ship, were captured on Taiwan after being abandoned by their British officers and were executed by local Qing officials.[50] Portuguese, Indian, American, and European crew members of the Ann, another British ship, were shipwrecked in Tamsui's vicinity in March 1842, captured, and then executed by the Chinese.[51][52] At Tainan 197 of the Nerbudda and Ann's crew were killed and due to causes related to imprisonment 87 others died.[53] | END ID: 425

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 426 | TITLE: The Pillars of the Earth | CONTENT: After many years, Kingsbridge cathedral is completed. Waleran still seeks to ruin Philip, and accuses him of fornication by claiming that Jonathan, now a well liked and committed monk, is Philip's son. With Philip's conviction certain due to a lack of evidence proving his innocence, Jack and Jonathan attempt to figure out the identity of the latter's father, both being unaware that he is Tom's son. They discover the truth when Jonathan recalls that he had been found near the monastery cell that Philip once ran, a fact that had previously been unknown to Jack, who then remembers seeing the baby Jonathan lying on his mother's grave. The two of them manage to convince Ellen, who has remained bitter towards Philip for his role in splitting up her and Tom, to testify on his behalf. | END ID: 426

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 427 | TITLE: You're Not the One | CONTENT: During a July 2013 interview with Glamour, Ferreira announced that the track would serve as the lead single from an upcoming extended playâ€”which ultimately became her debut album Night Time, My Time.[3] The single cover for "You're Not the One" was revealed on September 9, 2013, and the song itself was released 15 days later, on September 24.[4][5] An official remix of the track by producer Cid Rim was released online in March 2014;[6] a commercial release of that remix, packaged with three others in a digital EP format, followed in June.[7] In 2015, the Blood Diamonds remix of the song was featured in the commercial for Jimmy Choo's fragrance Illicit.[8] | END ID: 427

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 428 | TITLE: List of gridiron football rules | CONTENT: Similarly to association football, the game begins with a coin toss to determine which team will kick off to begin the game and which goal each team will defend.[2] The options are presented again to start the second half; the choices for the first half do not automatically determine the start of the second half (i.e. it is possible for the same team to kick off both halves).[3] The referee conducts the coin toss with the captains (or sometimes coaches) of the opposing teams. The team that wins the coin toss has three options:[2] | END ID: 428

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 429 | TITLE: Cooling tower | CONTENT: Cooling towers originated in the 19th century through the development of condensers for use with the steam engine.[2] Condensers use relatively cool water, via various means, to condense the steam coming out of the cylinders or turbines. This reduces the back pressure, which in turn reduces the steam consumption, and thus the fuel consumption, while at the same time increasing power and recycling boiler-water.[3] However the condensers require an ample supply of cooling water, without which they are impractical.[4][5] The consumption of cooling water by inland processing and power plants is estimated to reduce power availability for the majority of thermal power plants by 2040â€“2069.[6] While water usage is not an issue with marine engines, it forms a significant limitation for many land-based systems. | END ID: 429

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 430 | TITLE: Madonna (entertainer) | CONTENT: In 1992, Madonna starred in A League of Their Own as Mae Mordabito, a baseball player on an all-women's team. It reached number one on the box-office and became the tenth highest-grossing film of the year in the U.S.[91] She recorded the film's theme song, "This Used to Be My Playground", which became her tenth Hot 100 number-one hit, the most by any female artist at the time.[43] The same year, she founded her own entertainment company, Maverick, consisting of a record company (Maverick Records), a film production company (Maverick Films), and associated music publishing, television broadcasting, book publishing and merchandising divisions. The deal was a joint venture with Time Warner and paid Madonna an advance of $60 million. It gave her 20% royalties from the music proceedings, the highest rate in the industry at the time, equaled only by Michael Jackson's royalty rate established a year earlier with Sony.[92] | END ID: 430

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 431 | TITLE: Alkane | CONTENT: The density of the alkanes usually increases with the number of carbon atoms but remains less than that of water. Hence, alkanes form the upper layer in an alkaneâ€“water mixture. | END ID: 431

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 432 | TITLE: Jason Paige | CONTENT: Jason Paige (born January 6, 1969) is an American singer, writer, record producer, stage, film, and television actor. Paige is best known for singing the first theme song for the English version of the Pokémon television series. | END ID: 432

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 433 | TITLE: Artificial ventilation | CONTENT: The 1856 works of English physician and physiologist Marshall Hall recommended against using any type of bellows/positive pressure ventilation, views that held sway for several decades.[13] A common method of external manual manipulation, introduced in 1858, was the "Silvester Method" invented by Dr. Henry Robert Silvester in which a patient is laid on their back and their arms are raised above their head to aid inhalation and then pressed against their chest to aid exhalation. | END ID: 433

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 434 | TITLE: Hubble's law | CONTENT: where Di{\displaystyle D_{i}}

is the distance at which a galaxy is first measured at and 

t{\displaystyle t}

is the time since said measurement, thus the velocity of a galaxy can be expressed as 

v=D−Dit{\displaystyle v={\frac {D-D_{i}}{t}}}

. Solving the integral subsequently yields 

D=eH0t⋅Di{\displaystyle D=e^{H_{0}t}\cdot D_{i}} 

, with the implication that the comoving distance between two galaxies increases exponentially as time goes on. This lines up with current observations, however; it has not been shown for derivatives of distance above acceleration. | END ID: 434

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 435 | TITLE: List of 90210 characters | CONTENT: Constance Tate-Duncan, played by Maeve Quinlan in seasons one to three, is Adrianna's overbearing mother and a former actress, whose constant pressure has driven her daughter to drugs. She also has serious financial problems and depends on Adrianna to bring in money from acting gigs. | END ID: 435

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 436 | TITLE: State ownership | CONTENT: In Neoclassical economic theory, the desirability of state ownership has been studied using contract theory. According to the property rights approach based on incomplete contracting (developed by Oliver Hart and his co-authors), ownership matters because it determines what happens in contingencies that were not considered in prevailing contracts.[9] The work by Hart, Shleifer and Vishny (1997) is the leading application of the property rights approach to the question whether state ownership or private ownership is desirable.[10] In their model, the government and a private firm can invest to improve the quality of a public good and to reduce its production costs. It turns out that private ownership results in strong incentives to reduce costs, but it may also lead to poor quality. Hence, depending on the available investment technologies, there are situations in which state ownership is better. The Hart-Shleifer-Vishny theory has been extended in many directions. For instance, some authors have also considered mixed forms of private ownership and state ownership.[11] Moreover, the Hart-Shleifer-Vishny model assumes that the private party derives no utility from provision of the public good. Besley and Ghatak (2001) have shown that if the private party (a non-governmental organization) cares about the public good, then the party with the larger valuation of the public good should always be the owner, regardless of the parties' investment technologies.[12] Yet, more recently some authors have shown that the investment technology also matters in the Besley-Ghatak framework if an investing party is indispensable[13] or if there are bargaining frictions between the government and the private party.[14] | END ID: 436

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 437 | TITLE: Virgo (constellation) | CONTENT: IC 1101 is a supergiant elliptical galaxy in the Abell 2029 galaxy cluster located about 7025101229816056614♠1.07 Gly from Earth. At the diameter of 5.5 million light years, or more than 50 times the size of the Milky Way, it was the largest known galaxy in the universe. | END ID: 437

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 438 | TITLE: Easter Bunny | CONTENT: A chocolate Easter Bunny | END ID: 438

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 439 | TITLE: List of Flashpoint episodes | CONTENT: The fourth season of Flashpoint premiered on July 8, 2011. In the U.S., the show moved from CBS to Ion Television after "Shockwave."[49] | END ID: 439

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 440 | TITLE: Mount Everest in 2017 | CONTENT: A trekker to base camp died of altitude sickness in March.[72] | END ID: 440

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 441 | TITLE: Eartha Kitt | CONTENT: In 1984, she returned to the music charts with a disco song titled "Where Is My Man", the first certified gold record of her career. "Where Is My Man" reached the Top 40 on the UK Singles Chart, where it peaked at No. 36;[18] the song became a standard in discos and dance clubs of the time and made the Top 10 on the US Billboard dance chart, where it reached No. 7.[19] The single was followed by the album I Love Men on the Record Shack label. Kitt found new audiences in nightclubs across the UK and the United States, including a whole new generation of gay male fans, and she responded by frequently giving benefit performances in support of HIV/AIDS organizations. Her 1989 follow-up hit "Cha-Cha Heels" (featuring Bronski Beat), which was originally intended to be recorded by Divine, received a positive response from UK dance clubs and reached No. 32 in the charts in that country. In 1988 Eartha Kitt replaced Dolores Gray in the West End production of Stephen Sondheims Follies as Carlotta, receiving standing ovations every night for her rendition of "I'm Still Here" at the end of act 1. She went on to perform her own one-woman show at The Shaftesbury Theatre to sell-out houses for three weeks from 18 March 1989 after Follies closed. | END ID: 441

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 442 | TITLE: Plague of Athens | CONTENT: All the birds and beasts that prey upon human bodies, either abstained from touching them (though there were many lying unburied), or died after tasting them. In proof of this, it was noticed that birds of this kind actually disappeared; they were not about the bodies, or indeed to be seen at all.[9] | END ID: 442

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 443 | TITLE: J. Wellington Wimpy | CONTENT: Hamburgers are Wimpy's all-time favorite food, and he is usually seen carrying or eating one or more at a time – e.g., in Popeye the Sailor Meets Sindbad the Sailor he is seen grinding meat or eating burgers almost the entire time – however, he is usually too cheap to pay for them himself. A recurring joke involves Wimpy's attempts to con other patrons of the diner into buying his meal for him. His best-known catchphrase started in 1931 as, "Cook me up a hamburger. I'll pay you Thursday." In 1932, this then became the famous, "I'll gladly pay you Tuesday for a hamburger today."[5] The phrase was also slightly altered in the episode "Spree Lunch" to "I'll have a hamburger, for which I will gladly pay you Tuesday." This phrase is now commonly used to illustrate financial irresponsibility[6][7][8] and still appears in modern comedies such as The Drew Carey Show and The Office. The initial part of the phrase was even the title of Episode 6 of the fourth season of Cheers "I'll Gladly Pay You Tuesday." | END ID: 443

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 444 | TITLE: United States Senate | CONTENT: The Senate uses committees (and their subcommittees) for a variety of purposes, including the review of bills and the oversight of the executive branch. Formally, the whole Senate appoints committee members. In practice, however, the choice of members is made by the political parties. Generally, each party honors the preferences of individual senators, giving priority based on seniority. Each party is allocated seats on committees in proportion to its overall strength. | END ID: 444

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 445 | TITLE: GNU/Linux naming controversy | CONTENT: Proponents of the term GNU/Linux note that GNU alone would be just as good a name for GNU variants which combine the GNU operating system software with software from other sources.[5] GNU/Linux is a term promoted by the Free Software Foundation (FSF) and its founder Richard Stallman.[6] Proponents call for the correction of the more extended term, on the grounds that it doesn't give credit to the major contributor and the associated free software philosophy.[1][7] GNU is a longstanding project begun in 1984 to develop a free operating system. It is argued that when the Linux kernel was independently created in 1991, it merely provided a substantial missing piece.[6] Several distributions employ the FSF-endorsed name, such as Debian,[8] Trisquel[9] and Parabola GNU/Linux-libre.[10] | END ID: 445

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 446 | TITLE: United States Agency for International Development | CONTENT: After 1945, many newly independent countries needed assistance to relieve the chronic deprivation afflicting their low-income populations. USAID and its predecessor agencies have continuously provided poverty relief in many forms, including assistance to public health and education services targeted at the poorest. USAID has also helped manage food aid provided by the U.S. Department of Agriculture. In addition, USAID provides funding to NGOs to supplement private donations in relieving chronic poverty. | END ID: 446

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 447 | TITLE: Pokémon Mystery Dungeon: Blue Rescue Team and Red Rescue Team | CONTENT: By the end of 2006, PokÃ©mon Mystery Dungeon: Blue Rescue Team sold over 761,000 copies in Japan, while Red Rescue Team sold just over 715,000 copies.[13] As of July 25, 2007, Blue Rescue Team has sold 3.08 million copies worldwide while Red Rescue Team sold 2.20 million copies by March 31.[14][15] | END ID: 447

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 448 | TITLE: V6 News | CONTENT: Hamaraa Hyderabad It is a local news and events program pertaining to the city of Hyderabad. | END ID: 448

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 449 | TITLE: Center pivot irrigation | CONTENT: In the United States early settlers of the semiarid High Plains were plagued by crop failures due to cycles of drought, culminating in the disastrous Dust Bowl of the 1930s. Only after World War II when center pivot irrigation became available did the land mass of the High Plains aquifer system transform into one of the most agriculturally productive regions in the world. | END ID: 449

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 450 | TITLE: Osteoarthritis | CONTENT: In the United States, there were approximately 964,000 hospitalizations for osteoarthritis in 2011, a rate of 31 stays per 10,000 population.[155] With an aggregate cost of $14.8 billion ($15,400 per stay), it was the second-most expensive condition seen in U.S. hospital stays in 2011. By payer, it was the second-most costly condition billed to Medicare and private insurance.[156][157] | END ID: 450

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 451 | TITLE: 2009 Nobel Peace Prize | CONTENT: Nobel laureate and former U.S. Vice President Al Gore called the award "extremely well deserved",[52] Obama received congratulations and kind words from other elected officials, such as from House Speaker Nancy Pelosi and former rival, Senator John McCain, who said "As Americans, we're proud when our president receives an award of that prestigious category".[53] RNC chairman Michael Steele discussed his disapproval of the award in a fund-raising letter, writing, "the Democrats and their international leftist allies want America made subservient to the agenda of global redistribution and control."[54] | END ID: 451

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 452 | TITLE: Animas River | CONTENT: The Animas-La Plata Water Project was completed in 2015. The project pumps water over a low pass to fill a reservoir, Lake Nighthorse, in Ridges Basin to satisfy Southern Ute tribal water rights claims associated with the Colorado Ute Settlement Act amendments of 2000.[2] | END ID: 452

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 453 | TITLE: Chattanooga Choo Choo | CONTENT: Today, trains have a pride of place in Chattanooga's former Terminal Station. Once owned and operated by the Southern Railway, the station was saved from demolition after the withdrawal of passenger rail service in the early 1970s, and it is now part of a 30-acre (12-hectare) resort complex, including the Chattanooga Choo Choo Hotel, and numerous historical railway exhibits. Hotel guests can stay in half of a restored passenger railway car. Dining at the complex includes the Gardens restaurant in the Terminal Station itself, The Station House (which is housed in a former baggage storage room and known for its singing waitstaff) and the "Dinner in the Diner" which is housed in a restored 1941 Class A dining car. The music venue "Track29" is also on the grounds of the Chattanooga Choo Choo hotel in the building that formerly housed the city's only ice rink at the back of the property. The city's other historic station, Union Station, parts of which predated the Civil War, was demolished in 1973; the site is now an office building formerly housing the corporate offices of the Krystal restaurant chain (the restaurant chain offices have since relocated to Atlanta, Georgia). In addition to the railroad exhibits at "the Choo Choo", there are further exhibits at Tennessee Valley Railroad Museum, in east Chattanooga. | END ID: 453

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 454 | TITLE: Invention of radio | CONTENT: Oliver Heaviside, later reformulated Maxwell's original equations into the set of four vector equations that are generally known today as Maxwell's equations.[33] Neither Maxwell nor Heaviside transmitted or received radio waves; however, their equations for electromagnetic fields established principles for radio design, and remain the standard expression of classical electromagnetism. | END ID: 454

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 455 | TITLE: Portal:National Football League/Did you know | CONTENT: Portal:National Football League/Did you know/17 | END ID: 455

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 456 | TITLE: United Nations Convention on the Law of the Sea | CONTENT: Part XI of the Convention provides for a regime relating to minerals on the seabed outside any state's territorial waters or EEZ (Exclusive Economic Zones). It establishes an International Seabed Authority (ISA) to authorize seabed exploration and mining and collect and distribute the seabed mining royalty. | END ID: 456

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 457 | TITLE: Endangered Species Act of 1973 | CONTENT: Some have argued that the recovery of DDT-threatened species such as the bald eagle, brown pelican and peregrine falcon should be attributed to the 1972 ban on DDT by the EPA. rather than the Endangered Species Act, however, the listing of these species as endangered was a substantial cause of Congress instituting the ban and many non-DDT oriented actions were taken on their behalf under the Endangered Species Act (i.e. captive breeding, habitat protection, and protection from disturbance). | END ID: 457

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 458 | TITLE: Doraemon: Nobita's Three Visionary Swordsmen | CONTENT: Nobita and Doraemon meet with Shizuka. All of them got covered in sand due to falling of sand.Doraemon suggests that in this form dragon can not identify them. So they move to inner cave of the dragon. Nobita screamed to see Gian and Suneo in stoned form. Dragon hears it and attack Nobita with his fire, but he protect himself with his sword. He cuts the dragon's mustaches with the sword, causing the dragon to faint. When he was at the verge of finishing the dragon, he stops and gets away. All of them agree with Nobita. Dragon regains conscious, he tells Nobita that he does not want to turn any one to stone, he just wants to protect himself. He lets them bath in his perspiration which will grant him one more time to live. He also turns Gian and Suneo into normal humans. | END ID: 458

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 459 | TITLE: Grand Canyon | CONTENT: The Upper Sonoran Life Zone includes most of the inner canyon and South Rim at elevations from 3,500 to 7,000 feet (1,100 to 2,100 m).[55] This zone is generally dominated by blackbrush, sagebrush, and pinyon-juniper woodlands. Elevations of 3,500 to 4,000 feet (1,100 to 1,200 m) are in the Mojave Desert Scrub community of the Upper Sonoran. This community is dominated by the four-winged saltbush and creosote bush; other important plants include Utah agave, narrowleaf mesquite, ratany, catclaw acacia, and various cacti species.[55] | END ID: 459

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 460 | TITLE: Shravanabelagola | CONTENT: The sacred places are spread over two hills, Chandragiri and Vindyagiri, also among the village at the foothill. | END ID: 460

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 461 | TITLE: University of Texas at Arlington | CONTENT: The College of Nursing and Health Innovation is a nationally recognized program, the largest nursing program in Texas, and one of the five largest public nursing programs in the U.S. with over 8,000 nursing students in the BSN, RN‐to‐BSN, MSN, Post‐MSN, DNP, and PhD programs. | END ID: 461

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 462 | TITLE: Alan Kulwicki | CONTENT: It's been a long road and it's taken a lot of hard work to get here, but this has made it all worthwhile. When you work for something so hard for so long, you wonder if it's going to be worth all of the anticipation. Believe me, it certainly was. And what do you think of my Polish victory lap? There will never be another first win and you know, everybody sprays champagne or stands up on the car. I wanted to do something different for the fans.[36] | END ID: 462

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 463 | TITLE: Dennis Hopper | CONTENT: Hopper's fascination with art began with painting lessons at the Nelson-Atkins Museum while still a child in Kansas City, Missouri.[31] Early in his career, he painted and wrote poetry, though many of his works were destroyed in a 1961 fire that burned scores of homes, including his, on Stone Canyon Road[32] in Bel Air.[33] His painting style ranges from abstract impressionism to photorealism and often includes references to his cinematic work and to other artists.[1][34] | END ID: 463

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 464 | TITLE: The Magician's Nephew | CONTENT: The sacred Garden in the west of the Narnian world is surrounded by a "high wall of green turf" with branches of trees overhanging it, and "high gates of gold, fast shut, facing due east", which must be the only entrance because the travellers "walked nearly all the way round it" before they found them. In all these points Lewis echoes Milton's description of Eden in Paradise Lost: | END ID: 464

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 465 | TITLE: List of FIFA World Cup records | CONTENT: Fewest goals conceded, one tournament | END ID: 465

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 466 | TITLE: Dewey: The Small-Town Library Cat Who Touched the World | CONTENT: The School Library Journal (SLJ) said, | END ID: 466

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 467 | TITLE: Justified (TV series) | CONTENT: Throughout its run, Justified received largely positive reviews from critics. On the review aggregation website Metacritic, the first season scored 80/100, based on reviews from 27 critics.[24] The second season scored 91/100, based on reviews from 12 critics.[25] The third season scored 89/100, based on reviews from 14 critics.[26] The fourth season scored 90/100, based on reviews from 14 critics.[27] The fifth season scored 84/100, based on reviews from 14 critics.[28] The sixth season scored 89/100, based on reviews from 11 critics.[29] All seasons' scores but the first, which was one point short, indicate "universal acclaim."[24][25][26][27][28][29] | END ID: 467

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 468 | TITLE: Foreign exchange market | CONTENT: Non-bank foreign exchange companies offer currency exchange and international payments to private individuals and companies. These are also known as "foreign exchange brokers" but are distinct in that they do not offer speculative trading but rather currency exchange with payments (i.e., there is usually a physical delivery of currency to a bank account). | END ID: 468

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 469 | TITLE: Stanley Cup | CONTENT: As the prestige of winning the Cup grew, so did the need to attract top players. Only nine months after winning the Cup, in March 1906, the Montreal Wanderers pushed through a resolution at the annual meeting of the Eastern Canada Amateur Hockey Association (ECAHA) to allow professional players to play alongside amateurs. Because the ECAHA was the top hockey league in Canada at the time, the Cup trustees agreed to open the challenges to professional teams.[25] The first professional competition came one month later during the Wanderers' two-game, total goals challenge series, which they won 17Â goals to 5.[26] | END ID: 469

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 470 | TITLE: Genetic drift | CONTENT: where T is the number of generations, Ne is the effective population size, and p is the initial frequency for the given allele. The result is the number of generations expected to pass before fixation occurs for a given allele in a population with given size (Ne) and allele frequency (p).[23] | END ID: 470

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 471 | TITLE: Amazon tax | CONTENT: In a 2012 editorial supporting tax equity, the Florida St. Petersburg Times wrote, "As long as Internet-only sellers such as Amazon.com can get away with not collecting state sales tax and effectively sell their products for at least 6 percent less, Florida merchants pay the price. It's past time for lawmakers to work toward a level playing field."[37] | END ID: 471

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 472 | TITLE: Time Enough at Last | CONTENT: "Time Enough at Last" was a ratings success in its initial airing and "became an instant classic".[12] It "remains one of the best-remembered and best-loved episodes of The Twilight Zone" according to Marc Zicree, author of The Twilight Zone Companion.[6] When a poll asked readers of Twilight Zone Magazine which episode of the series they remembered the most, "Time Enough at Last" was the most frequent response, with "To Serve Man" coming in a distant second.[13] In TV Land's presentation of TV Guide's "100 Most Memorable Moments in Television", "Time Enough at Last" was ranked at #25.[14] 
In an interview, Serling cited "Time Enough at Last" as one of his two favorites from the entire series. (The other episode was "The Invaders", with Agnes Moorehead.)[15] | END ID: 472

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 473 | TITLE: Arguments for and against drug prohibition | CONTENT: The freedom of choice of those addicted to a drug is also questioned, recognizing that addiction is defined as compulsive by its very nature[153] and that addictions in and of themselves curb individual freedom. Likewise, the proposal that addictive drugs should be legalized, regulated and opened to "free market dynamics" is immediately belied by the recognition that the drug market for an addict is no longer a free market â€“ it is clear that they will pay any price when needing their drug.[1] | END ID: 473

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 474 | TITLE: Hydra (comics) | CONTENT: Meanwhile, after having destroyed Hydra's undersea headquarters, Ichor, due to its having been infiltrated by the Skrull invasion force, Von Strucker rebuilt Hydra from the ground up, and after his discovery that Fury had learned the truth, reconvened the other main heads of Hydra: Viper, Madame Hydra, Kraken, and The Hive, as well as resurrecting The Gorgon for the purpose of showing Hydra's "True self" to the world.[7] | END ID: 474

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 475 | TITLE: Anxiety disorder | CONTENT: Self-help books can contribute to the treatment of people with anxiety disorders.[93] | END ID: 475

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 476 | TITLE: National Statistical Office of Malawi | CONTENT: Situated off Chimbiya Road, Zomba, the NSO Headquarters contains the offices of the Commissioner of Statistics and Deputy Commissioner of Statistics, together with the general administration, accounts and human resources departments. | END ID: 476

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 477 | TITLE: History of Turkey | CONTENT: Turkey was neutral in World War II (1939â€“45) but signed a treaty with Britain in October 1939 that said Britain would defend Turkey if Germany attacked it. An invasion was threatened in 1941 but did not happen and Ankara refused German requests to allow troops to cross its borders into Syria or the USSR. Germany had been its largest trading partner before the war, and Turkey continued to do business with both sides. It purchased arms from both sides. The Allies tried to stop German purchases of chrome (used in making better steel). Starting in 1942 the Allies provided military aid.  The Turkish leaders conferred with Roosevelt and Churchill at the Cairo Conference in November, 1943, and promised to enter the war. By August 1944, with Germany nearing defeat, Turkey broke off relations. In February 1945, it declared war on Germany and Japan, a symbolic move that allowed Turkey to join the nascent United Nations.[44][45] | END ID: 477

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 478 | TITLE: Ninja Re Bang Bang | CONTENT: Most references are taken from Natari.jp, which also shows pictures of the filming.[8] | END ID: 478

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 479 | TITLE: Names of God in Islam | CONTENT: ذو الجلال ولإكرام | END ID: 479

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 480 | TITLE: Geotagging | CONTENT: Where the above methods are in use, their coordinates may differ from those specified by the photo's internal Exif data, for example because of a correction or a difference between the camera's location and the subject's. | END ID: 480

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 481 | TITLE: Chevrolet El Camino | CONTENT: General Motors-Holden's manufactured and marketed coupé utility models in Australia commencing in 1935. GMH continued to offer a Chevrolet coupé utility until 1952. Rebadged Holden coupé utilities, including later Commodore-based models, were sold as the Chevrolet El Camino and Chevrolet Lumina in South Africa and the Middle East. Holden produced a Commodore-based coupé utility in Australia as the Holden Ute until 2017, when Holden's Elizabeth factory closed down. | END ID: 481

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 482 | TITLE: Multiple myeloma | CONTENT: The risk of multiple myeloma can be reduced by maintaining a normal body weight.[42] | END ID: 482

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 483 | TITLE: Imperial City, Beijing | CONTENT: In 1399, Zhu Di launched a coup d'Ã©tat and ascended to the throne to become Yongle Emperor in 1402. In 1403, the name of Beiping was changed to Beijing (literally "the Northern Capital"), and in 1406 a plan was drafted to move the capital to Beijing. | END ID: 483

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 484 | TITLE: Battle of Cambrai (1917) | CONTENT: The German counter-attack showed the effectiveness of artillery, trench mortars and evolving stormtrooper tactics, adopted from a pattern introduced by General Hutier against the Russians.[38][page needed][36] From the German perspective, questions arose regarding battlefield supply beyond railheads and the suitability of the MG 08 machine gun for rapid movement.[39][page needed] By the end of the battle, the British retained some of the ground captured in the north and the Germans a smaller amount taken in the south. The British conducted several investigations, including a Court of Enquiry.[38][page needed] | END ID: 484

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 485 | TITLE: CompStat | CONTENT: Compstat offers a dynamic approach to crime reduction, quality of life improvement, and personnel and resource management, whereby ranking police department executives identify spikes in crimes using comparative statistics and address those spikes through the use of targeted enforcement. To this end, Compstat includes four generally recognized components: timely and accurate information or intelligence, rapid deployment of resources, effective tactics, and relentless follow-up. However, Compstat can be expanded and tweaked depending on specific department needs. Originally, it was modeled after the broken windows theory, whereby minor crimes would be addressed in order to reduce major crimes. However, over time, its use evolved into a system whereby productivity was measured and individuals were held accountable for spikes in crime. Commercial entities began producing turnkey packages (including computer systems, software, mobile devices, and other implements) assembled under the heading of CompStat. For example, Geographic Information Systems allow departments to map crime or other types of data, to aid in identifying and solving problems in their assigned area. | END ID: 485

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 486 | TITLE: Public Company Accounting Oversight Board | CONTENT: The Board's Office of the Chief Auditor advises the Board on the establishment of auditing and related professional practice standards.
[4] | END ID: 486

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 487 | TITLE: Ozarks | CONTENT: There are two mountain ranges within the Ozarks: the Boston Mountains of Arkansas and the St. Francois Mountains of Missouri. Buffalo Lookout, the highest point in the Ozarks, is located in the Boston Mountains. Geologically, the area is a broad dome with the exposed core in the St. Francois Mountains. The Ozarks cover nearly 47,000 square miles, making it the most extensive highland region between the Appalachians and Rockies. Together with the Ouachita Mountains, the area is known as the U.S. Interior Highlands. | END ID: 487

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 488 | TITLE: Counterintelligence | CONTENT: In Country C, Service A surveys the intelligence terrain through the eyes of Service B (a species of mirror-reading) and selects those citizens whose access to sources and other qualifications make them most attractive to B. Service A officers, posing as service B officers, recruit the citizens of country C. At some point, service A then exposes these individuals, and complains to country C that country B is subverting its citizens. | END ID: 488

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 489 | TITLE: Leaving Certificate (Ireland) | CONTENT: Subjects are examined through a number of methods. These will include at least one written paper (English, Mathematics, Irish and some of the optional courses contain two written papers). | END ID: 489

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 490 | TITLE: 101 Dalmatians (1996 film) | CONTENT: American video game designer Roger Dearly (Jeff Daniels) lives with his pet dalmatian Pongo in London. One day, Pongo sets his eyes on a beautiful female dalmatian named Perdy. After a frantic chase through the streets of London that ends in St. James's Park, Roger discovers that Pongo likes Perdy. Her owner, Anita Campbell-Green (Joely Richardson) falls in love with Roger when they meet. They both fall into the lake as a result of their dogs chasing each other, but they return to Roger's home and Anita accepts his proposal. They get married along with Perdita and Pongo. Anita works as a fashion designer at the House of de Vil. Her boss, the pampered and very glamorous Cruella de Vil (Glenn Close), has a deep passion for fur, going so far as to have a taxidermist, Mr Skinner, skin a white tiger at the London Zoo to make it into a rug for her. Anita, inspired by her dalmatian, designs a coat made with spotted fur. Cruella is intrigued by the idea of making garments out of actual dalmatians, and finds it amusing that it would seem as if she was wearing Anita's dog. | END ID: 490

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 491 | TITLE: Discrimination | CONTENT: Transgender individuals, whether male-to-female, female-to-male, or genderqueer, often experience transphobic problems that often lead to dismissals, underachievement, difficulty in finding a job, social isolation, and, occasionally, violent attacks against them. Nevertheless, the problem of gender discrimination does not stop at transgender individuals or with women. Men are often the victim in certain areas of employment as men begin to seek work in office and childcare settings traditionally perceived as "women's jobs". One such situation seems to be evident in a recent case concerning alleged YMCA discrimination and a Federal Court Case in Texas.[66] The case actually involves alleged discrimination against both men and black people in childcare, even when they pass the same strict background tests and other standards of employment. It is currently being contended in federal court, as of fall 2009. | END ID: 491

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 492 | TITLE: Battle of Shiloh | CONTENT: Maj. Gen. Ulysses S. Grant's Army of the Tennessee of 44,895[9][24] men consisted of six divisions: | END ID: 492

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 493 | TITLE: Juliette Barnes | CONTENT: 2014: Inside The Dream North American Tour | END ID: 493

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 494 | TITLE: Mount Whitney | CONTENT: In 1891, the United States Geological Survey's Board on Geographic Names decided to recognize the earlier name Mount Whitney. Despite losing out on their preferred name, residents of Lone Pine financed the first trail to the summit, engineered by Gustave Marsh, and completed on July 22, 1904. Just four days later, the new trail enabled the first recorded death on Whitney. Having hiked the trail, U.S. Bureau of Fisheries employee Byrd Surby was struck and killed by lightning while eating lunch on the exposed summit. In response to this event, Marsh began work on the stone hut that would become the Smithsonian Institution Shelter, and completed it in 1909.[27] | END ID: 494

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 495 | TITLE: Geography of Iran | CONTENT: Rain forest in Gilan | END ID: 495

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 496 | TITLE: Call Me by Your Name (film) | CONTENT: The film was shot primarily in Crema,[14][25] and in the nearby villages of Pandino and Moscazzano. It was shot during an unexpected historic rainstorm in Italy, described by the weather reports as a "once-in-century rain."[48] The pre-production in Crema was fast:[10] a search for extras began there in March and April.[55][56] Scenes from Pandino and Moscazzano were filmed from May 17,[57][58] before moving to Crema on June 1.[59] Additional outdoor scenes were shot at the Palazzo Albergoni on December 4, 2016.[60][61] Several historical locations in the surrounding streets in Crema and Pandino were chosen during production, including the Crema Cathedral.[57][62] Businesses received compensation for financial losses caused by the closure, scheduled for May 30 and 31.[63] Two days' filming at the cathedral were postponed due to the weather.[62] Production in Crema cost €18,000,[64] with a promotion campaign that cost €7,500.[65] Filming also took place at the Grottoes of Catullus by Lake Garda in Sirmione,[66] the Cascate del Serio in Bergamo,[67] and in two small towns in the immediate vicinity of Crema: Montodine and Ripalta.[8][60] | END ID: 496

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 497 | TITLE: Lorraine Chase | CONTENT: As well as her various television credits, Chase has also appeared in a variety of stage productions, including pantomime, comedy and drama. Her first acting role following her Campari advertisements in the 1970s was in a play, The Undertaking, starring Kenneth Williams.[6] Since then she has appeared in stage productions of Pygmalion, Little Shop of Horrors, Me and My Girl, Tea For Two, and Run For Your Wife. In 2007, she toured the UK in a production of the thriller Dead Guilty.[7] | END ID: 497

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 498 | TITLE: Politics of the United States | CONTENT: The United States is a federal republic in which the president, Congress, and federal courts share powers reserved to the national government according to its Constitution. The federal government shares sovereignty with the state governments. | END ID: 498

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 499 | TITLE: Per Manum | CONTENT: "Per Manum" featured the appearance of David Duchovny as Fox Mulder in various flashbacks. After settling his contract dispute with Fox, Duchovny quit full-time participation in the show after the seventh season.[3] In order to explain Mulder's absence, Duchovny's character was abducted by aliens in the seventh season finale, "Requiem". After several rounds of contractual discussions, Duchovny agreed to return for a total of 11 season eight episodes.[4] "Per Manum" marked the fourth appearance of Duchovny in the eighth season; he had previously appeared in opening episodes of the season, "Within" and "Without" as well as the eleventh episode "The Gift".[5][6][7] Series creator Chris Carter later argued that Mulder's absences from the series did not affect the characterization, noting that "there are characters who can be powerful as absent centers, as Mulder was through the eighth and ninth seasons."[8] | END ID: 499

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 500 | TITLE: Catholic Church in Australia | CONTENT: In 2001, in Rome, Pope John Paul II apologised to Aborigines and other indigenous people in Oceania for past injustices by the church: Aware of the shameful injustices done to indigenous peoples in Oceania, the Synod Fathers apologised unreservedly for the part played in these by members of the church, especially where children were forcibly separated from their families. Church leaders in Australia called on the Australian government to offer a similar apology.[72] | END ID: 500

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 501 | TITLE: West Bromwich Albion F.C. | CONTENT: Football League First Division (old), Premier League (modern) (1) | END ID: 501

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 502 | TITLE: Suez Crisis | CONTENT: After Suez, Cyprus, Aden, and Iraq became the main bases for the British in the region while the French concentrated their forces at Bizerte and Beirut. UNEF was placed in the Sinai (on Egyptian territory only) with the express purpose of maintaining the cease-fire. While it was effective in preventing the small-scale warfare that prevailed before 1956 and after 1967, budgetary cutbacks and changing needs had seen the force shrink to 3,378 by 1967. | END ID: 502

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 503 | TITLE: Human sacrifice in Aztec culture | CONTENT: To appease Huehueteotl, the fire god and a senior deity, the Aztecs had a ceremony where they prepared a large feast, at the end of which they would burn captives; before they died they would be taken from the fire and their hearts would be cut out. Motolinía and Sahagún reported that the Aztecs believed that if they did not placate Huehueteotl, a plague of fire would strike their city. The sacrifice was considered an offering to the deity.[27] | END ID: 503

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 504 | TITLE: Visual cortex | CONTENT: However, since neurons in V1 are also tuned to the direction and speed of motion, these early results left open the question of precisely what MT could do that V1 could not. Much work has been carried out on this region, as it appears to integrate local visual motion signals into the global motion of complex objects.[50] For example, lesion to the V5 leads to deficits in perceiving motion and processing of complex stimuli. It contains many neurons selective for the motion of complex visual features (line ends, corners). Microstimulation of a neuron located in the V5 affects the perception of motion. For example, if one finds a neuron with preference for upward motion in a monkey's V5 and stimulates it with an electrode, then the monkey becomes more likely to report 'upward' motion when presented with stimuli containing 'left' and 'right' as well as 'upward' components.[51] | END ID: 504

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 505 | TITLE: Stand by Your Ad provision | CONTENT: Campaigns have lamented that the seconds used for the candidates to approve the communication results in less time for them to communicate their message, increasing their costs of campaigning. One media adviser mentioned that the requirement reduced the number of positive spots that the producer can have.[17] Other candidates, however, regard it as an opportunity to affirm or encapsulate the theme of their message: "I'm Tom Kean, Jr. Together, we can break the back of corruption. That's why I approved this message." | END ID: 505

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 506 | TITLE: Intoxication defense | CONTENT: If a "specific intent" in either sense is required and there is clear evidence that the accused was too intoxicated to form the element subjectively, this fact is recognised as a defense unless the loss of control was part of the plan. But this is of little value to defendants since there are almost always offenses of basic intent that can be charged and/or the basic intent offenses are usually lesser included offenses and an alternative verdict can be delivered by judge or jury without the need for a separate charge. In English law, note the controversial Jaggard v Dickinson [1980] 3 All ER 716 which held that, for the purposes of the statutory defense of lawful excuse under s5 Criminal Damage Act 1971, a drunken belief will found the defense even though this allows drunkenness to negate basic intent. This is limited authority and does not affect the generality of the defense. | END ID: 506

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 507 | TITLE: Nanny McPhee | CONTENT: The film was theatrically released on 28 October 2005 in the UK and on 27 January 2006 in the USA by Universal Pictures and was released on DVD on 9 May 2006 by Universal Studios Home Entertainment. | END ID: 507

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 508 | TITLE: Wyatt Earp | CONTENT: Fred Dodge arrived on the scene. In a letter to Stuart Lake many years later, he recalled what he saw. | END ID: 508

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 509 | TITLE: Gannett Company | CONTENT: On May 14, 2014, Gannett announced the acquisition of six stations from the Texas-based London Broadcasting Company in a $215 million deal, including KCEN-TV (NBC) in Waco-Temple-Bryan, KYTX (CBS) in Tyler-Longview, KIII (ABC) in Corpus Christi, KBMT (ABC/NBC) in Beaumont-Port Arthur, KXVA (FOX) in Abilene-Sweetwater and KIDY (FOX) in San Angelo. The company's COO Phil Hurley will also join Gannett to continue his leadership role at the six stations.[30] The acquisition was completed on July 8, 2014; in total, Gannett stations now serve 83% of households in the state.[31] Post acquisition, Gannett now outright owns and operates their first Fox affiliates, KIDY & KXVA. | END ID: 509

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 510 | TITLE: Soap (TV series) | CONTENT: The Roman Catholic Church, led by its Los Angeles Archdiocese, also condemned the show and asked all American families to boycott it saying "ABC should be told that American Catholics and all Americans are not going to sit by and watch the networks have open season on Catholicism and morality. [Soap] is probably one of the most effective arguments for government censorship of TV that has yet come along."[10] In August, the Board of Rabbis of Southern California representing three branches of Judaism, joined the Catholic protest saying that the as-yet unaired show "reached a new low". | END ID: 510

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 511 | TITLE: New York Giants | CONTENT: The Giants were one of five teams that joined the NFL in 1925, and is the only one of that group still existing, as well as the league's longest-established team in the Northeastern United States. The team ranks third among all NFL franchises with eight NFL championship titles: four in the preâ€“Super Bowl era (1927, 1934, 1938, 1956) and four since the advent of the Super Bowl (Super Bowls XXI (1986), XXV (1990), XLII (2007), and XLVI (2011)), along with more championship appearances than any other team, with 19 overall appearances. Their championship tally is surpassed only by the Green Bay Packers (13) and Chicago Bears (9). Throughout their history, the Giants have featured 28 Hall of Fame players, including NFL Most Valuable Player (MVP) award winners Mel Hein, Frank Gifford, Y. A. Tittle, and Lawrence Taylor. | END ID: 511

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 512 | TITLE: Nicaragua | CONTENT: Without women in their parties,[25]:123 the Spanish conquerors took Nahua and Chorotega wives and partners, beginning the multiethnic mix of native and European stock now known as "mestizo", which constitutes the great majority of the population in western Nicaragua.[26] Many indigenous people died as a result of new infectious diseases, compounded by neglect by the Spaniards, who controlled their subsistence.[32] Furthermore, a large number of other natives were captured and transported to Panama and Peru between 1526 and 1540, where they were forced to perform slave labor.[22]:193[25]:104â€“105 | END ID: 512

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 513 | TITLE: Caliphate | CONTENT: In 1899 John Hay, U.S. Secretary of State, asked the American ambassador to Ottoman Turkey, Oscar Straus, to approach Sultan Abdul Hamid II to use his position as caliph to order the Tausūg people of the Sultanate of Sulu in the Philippines to submit to American suzerainty and American military rule; the Sultan obliged them and wrote the letter which was sent to Sulu via Mecca. As a result, the "Sulu Mohammedans ... refused to join the insurrectionists and had placed themselves under the control of our army, thereby recognizing American sovereignty."[37][37][38] | END ID: 513

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 514 | TITLE: The New York Times | CONTENT: In 1896, Adolph Ochs bought The New York Times, a money-losing newspaper, and formed the New York Times Company. The Ochs-Sulzberger family, one of the United States' newspaper dynasties, has owned The New York Times ever since.[37] The publisher went public on January 14, 1969, trading at $42 a share on the American Stock Exchange.[103] After this, the family continued to exert control through its ownership of the vast majority of Class B voting shares. Class A shareholders are permitted restrictive voting rights while Class B shareholders are allowed open voting rights. | END ID: 514

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 515 | TITLE: Wealth inequality in the United States | CONTENT: Distribution of net worth in the United States (2007).[18] The net wealth of many people in the lowest 20% is negative because of debt.[18] By 2014 the wealth gap deepened. | END ID: 515

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 516 | TITLE: Bridgewater Canal | CONTENT: Between 1849 and 1851 the competition between the Trustees and the railway companies intensified. Agreements and alliances were made and broken. Their major opponents were the London and North Western Railway and the Lancashire and Yorkshire Railway who reduced tariffs and took business away from the canals. For the first time the railways carried more trade between Liverpool and the towns of central Lancashire than the canals.[89] The value of the traffic carried by the Bridgewater Canal in 1851 was the lowest in the time it was administered by the Trustees.[56] In 1851 the Earl of Ellesmere hosted a visit to Manchester by Queen Victoria and Prince Albert. They stayed at Worsley Hall, with a view of the canal, and were given a trip between Patricroft railway station and Worsley Hall, on state barges. Large crowds had gathered to cheer the royal party, which apparently frightened the horses drawing the barge so much that they fell into the canal.[90] | END ID: 516

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 517 | TITLE: Poverty in Canada | CONTENT: Relative poverty measures, the most prominent being income distribution measures, also known as income inequality metrics, reveal information about disparities of income within a population. So, for instance, if a society becomes richer, even those in the bottom income bands may see their incomes rise as well. A measure which accounts for this rise, increasing with the average income of the society, is known as a "relative measure of poverty."[36] Relative poverty measures are considered by some to be the most useful for advanced industrial nations like Canada.[notes 3] According to a 2008 report by the Organisation for Economic Co-operation and Development (OECD), the rate of poverty in Canada, is among the highest of the OECD member nations, the world's wealthiest industrialized nations.[1] There is no official government definition and therefore, measure, for poverty in Canada. However, Dennis Raphael, author of Poverty in Canada: Implications for Health and Quality of Life[2][3] reported that the United Nations Development Program (UNDP), the United Nations Children's Fund (UNICEF), the Organisation for Economic Co-operation and Development (OECD) and Canadian poverty researchers[notes 4][4] find that relative poverty is the "most useful measure for ascertaining poverty rates in wealthy developed nations such as Canada."[1][5][6][7][8] | END ID: 517

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 518 | TITLE: List of Shameless (UK TV series) episodes | CONTENT: Carl arrives back on the Chatsworth as a qualified policeman, but he receives hostility from Frank and a surprise welcoming from Jamie. But Carl is determined not to go ahead with the training until he receives his father's blessing. Meanwhile, Jackson is fingered for falsely claiming benefits, resulting in the withdrawal of his teaching job, and his marriage is thrown into chaos. he joins forces with the Maguires to save the estate, hoping it will save his marriage, while Frank is forced to become a double agent to gather information on Kenaway's next move, but Kenaway is one step ahead and riots begin to occur. Can they prevent the council from destroying Chatsworth forever? Elsewhere, Kelly visits a client in the hospital and runs into his wife, sparking off a heartbreaking chain of events that will change her life forever, while a confused Gloria Meak (Angeline Ball) tries to remember who took advantage of her the night before. | END ID: 518

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 519 | TITLE: Sauron | CONTENT: With Sauron's assistance, the Elven-smiths forged the Rings of Power, which conferred great power upon their bearers. He then secretly forged the One Ring in the volcanic Mount Doom in Mordor. This "One Ring to rule them all" had the power to dominate the other Rings and enslave their wearers to Sauron's will. The Rings of Power were extremely potent, however; to create an instrument that could dominate even them, Sauron was forced to transfer a great part of his native power into it. Yet "while he wore it, his power on earth was actually enhanced".[30] | END ID: 519

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 520 | TITLE: Budget of the Government of Puerto Rico | CONTENT: Once approved the Department of Treasury disburses funds to the Office of Management and Budget which in turn disburses the funds to the respective agencies, all while the Puerto Rico Government Development Bank (the government's intergovernmental bank) manages all related banking affairs including those related to the government-owned corporations. | END ID: 520

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 521 | TITLE: South African insolvency law | CONTENT: Nobody wants the insolvent to be destitute. The insolvent, therefore, is allowed to follow any profession or occupation, and to enter into any contracts related thereto. The insolvent, however, requires the consent of the trustee in order to carry on the business of a trader or manufacturer. If the trustee refuses this permission, the insolvent may appeal to the Master. Why? Because of the disposition of assets: If your business is buying and selling, the trustee's work is made very difficult. | END ID: 521

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 522 | TITLE: Indian Statistical Institute | CONTENT: ISI offers two undergraduate programs, viz. Bachelor of Statistics (Honours) (B.Stat) and Bachelor of Mathematics (Honours) (B. Math),[21] seven graduate programs, viz. Master of Statistics (M. Stat), Master of Mathematics (M. Math), Master of Science in Quantitative Economics (MSQE), Master of Science in Library and Information Science (MSLIS), Master of Science in Quality Management Science (MSQMS), Master of Technology in Computer Science (MTech–CS) and Master of Technology in Quality, Reliability and Operations Research (MTech–QROR),[21] three PG Diploma programs, viz Post Graduate Diploma in Computer Applications (PGDCA)[22] and P.G. Diploma in Statistical Methods and Analytics and research fellowships towards obtaining a PhD degree.[21]. The third PG diploma program being in collaboration with IIM Calcutta and IIT Kharagpur - Post Graduate Diploma in Business Analytics (PGDBA)[23] with an aim to nurture and develop highly skilled business analytical professionals. | END ID: 522

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 523 | TITLE: Plitvice Lakes National Park | CONTENT: The name Plitvice was first mentioned in a written document in 1777 by Dominik Vukasović, the priest of Otočac.[7] This name was designated due to natural phenomena that have created the lakes. Nature formed shallow basins (Croatian pličina or plitvak, plitko means shallow), which have been filled with water. For centuries, water has changed the limestone and thus the landscape of this area. The emerging travertine barriers decelerated and retained the flowing water. These dams are continuously growing in height.[8] | END ID: 523

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 524 | TITLE: Whitney Dean | CONTENT: McGarty told Inside Soap that she hoped the storyline would have a positive impact, saying that she had done some research herself before filming, meeting teenage girls who had been exploited and hearing their experiences.[20] She said she felt honoured and privileged to be given the storyline.[21] | END ID: 524

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 525 | TITLE: Odyssey | CONTENT: The Odyssey (/ˈɒdəsi/;[1] Greek: Ὀδύσσεια Odýsseia, pronounced [o.dýs.sej.ja] in Classical Attic) is one of two major ancient Greek epic poems attributed to Homer. It is, in part, a sequel to the Iliad, the other work ascribed to Homer. The Odyssey is fundamental to the modern Western canon; it is the second-oldest extant work of Western literature, while the Iliad is the oldest. Scholars believe the Odyssey was composed near the end of the 8th century BC, somewhere in Ionia, the Greek coastal region of Anatolia.[2] | END ID: 525

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 526 | TITLE: Lewis antigen system | CONTENT: The Le gene encodes a fucosyltransferase that adds fucose to type 1 precursor substance (both free in serum and in secretions) to make the Le(a) antigen. The le gene is an amorph. The Lewis antigen produced on free type 1 precursor substance passively adsorbs onto the surfaces or red blood cells.[1] | END ID: 526

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 527 | TITLE: Environmental protection | CONTENT: In year 1972 was the first direct response from the federal government to address eminent health effects from environmental issues. It established the administrative organization of the Secretariat for the Improvement of the Environment (SubsecretarÃ­a para el Mejoramiento del Ambiente) in the Department of Health and Welfare. | END ID: 527

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 528 | TITLE: André the Giant Memorial Trophy | CONTENT: On the March 10, 2014 episode of Raw, WrestleMania XXX host Hulk Hogan announced that he was establishing the André the Giant Memorial Battle Royal in honor of André's legacy that would take place at the event on April 6, with the winner receiving the André the Giant Memorial Trophy (made in the likeness of André).[2] Cesaro would win the match after eliminating Big Show using a body slam similar to the body slam Hogan used on André at WrestleMania III.[4] | END ID: 528

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 529 | TITLE: Roman navy | CONTENT: By the time of the Notitia Dignitatum, the Classis Germanica has ceased to exist (it is last mentioned under Julian in 359),[136] most probably due to the collapse of the Rhine frontier after the Crossing of the Rhine by the barbarians in winter 405-406, and the Mauretanian and African fleets had been disbanded or taken over by the Vandals. | END ID: 529

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 530 | TITLE: Politics of Oklahoma | CONTENT: The Oklahoma Senate | END ID: 530

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 531 | TITLE: Behavioral neuroscience | CONTENT: Different manipulations have advantages and limitations. Neural tissue destroyed as a primary consequence of a surgery, electric shock or neurotoxin can confound the results so that the physical trauma masks changes in the fundamental neurophysiological processes of interest. For example, when using an electrolytic probe to create a purposeful lesion in a distinct region of the rat brain, surrounding tissue can be affected: so, a change in behavior exhibited by the experimental group post-surgery is to some degree a result of damage to surrounding neural tissue, rather than by a lesion of a distinct brain region.[30][31] Most genetic manipulation techniques are also considered permanent.[31] Temporary lesions can be achieved with advanced in genetic manipulations, for example, certain genes can now be switched on and off with diet.[31] Pharmacological manipulations also allow blocking of certain neurotransmitters temporarily as the function returns to its previous state after the drug has been metabolized.[31] | END ID: 531

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 532 | TITLE: Jacques-Louis David | CONTENT: The Death of Marat, perhaps David's most famous painting, has been called the Pietà of the revolution. Upon presenting the painting to the convention, he said "Citizens, the people were again calling for their friend; their desolate voice was heard: David, take up your brushes.., avenge Marat... I heard the voice of the people. I obeyed." David had to work quickly, but the result was a simple and powerful image. | END ID: 532

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 533 | TITLE: Joseph P. Kennedy Sr. | CONTENT: In 1941, Kennedy allowed surgeons to perform a lobotomy on his daughter Rosemary. Various reasons for the operation have been given, but it left her permanently incapacitated.[3][4][5] | END ID: 533

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 534 | TITLE: Philippines | CONTENT: Barasoain Church in Malolos, Bulacan, where the First Philippine Republic was founded | END ID: 534

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 535 | TITLE: Come On Eileen | CONTENT: Dexys Midnight Runners' CD compilations again omit the introduction and coda but use the unedited main section (4.06).[10] | END ID: 535

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 536 | TITLE: National Hockey League Players' Association | CONTENT: The NHLPA Executive Board terminated the employment of Saskin as Executive Director and General Counsel on May 10, 2007, following alleged acts of misconduct. Toronto employment lawyer Chris Paliare concluded Saskin and executive Ken Kim, beginning in September 2005 through January 2007, covertly accessed player email accounts. | END ID: 536

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 537 | TITLE: Waterfowl hunting | CONTENT: Many provinces in Canada and all states require hunters, including waterfowl hunters, to complete hunter safety courses before they can obtain a license.[4]
Waterfowl hunters fire short-range shotgun rounds into the air over often deserted bodies of water, so accidental injuries are rarer than in other hunting activities such as big game or deer hunting. | END ID: 537

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 538 | TITLE: A Change in Me | CONTENT: Since "A Change in Me" was introduced four years into Beauty and the Beast's run,[8] the song has not yet been included on any official English-language cast albums.[45] However, it has been recorded for the 2005 Manila, 2008 Madrid and 2009 Barcelona original cast recordings of the musical by various actresses in their native languages.[46] It has since been covered by several artists. Actress Susan Egan, who originated the role of Belle when the show premiered in 1994, had already long left the production by the time "A Change in Me" was introduced. She covered the song for her debut studio album So Far in 2002.[47][48] Arranged and produced by Craig Barna, Egan's version of "A Change in Me" was the first English-language studio recording of the song.[49] | END ID: 538

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 539 | TITLE: Irish Canadians | CONTENT: The family names, the features and colouring, the predominant Catholic religion, the prevalence of Irish music – even the dialect and accent of the people – are so reminiscent of rural Ireland that Irish author Tim Pat Coogan has described Newfoundland as "the most Irish place in the world outside of Ireland".[39] | END ID: 539

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 540 | TITLE: The Westing Game | CONTENT: Sunset Towers is a new apartment building on Lake Michigan, north of Milwaukee and just down the shore from the mansion owned by reclusive self-made millionaire Samuel W. Westing. (Despite the name, Sunset Towers faces east â€“ into the sunrise.) | END ID: 540

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 541 | TITLE: Charles Durning | CONTENT: For his numerous roles on television, he earned nine Emmy Award nominations. He also received Academy Award for Best Supporting Actor nominations for The Best Little Whorehouse in Texas in 1982 and To Be or Not to Be in 1983. He won a Golden Globe in 1990 for his supporting role in the television miniseries The Kennedys of Massachusetts, having had three previous nominations. That same year, he won a Tony Award for his performance as Big Daddy in Cat on a Hot Tin Roof. He received two Drama Desk Awards for his performances in That Championship Season and Third. | END ID: 541

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 542 | TITLE: Pay Commission | CONTENT: The Government of India has initiated the process to constitute the 7th Central Pay Commission along with finalisation of its Terms of Reference, the composition and the possible timeframe for submission of its Report.[18]
On 25 September 2013 then Finance Minister P Chidambaram announced that Prime Minister Manmohan Singh has approved the constitution of the 7th Pay Commission. Its recommendations are likely to be implemented with effect from 1 January 2016. 
Justice A.K Mathur will be heading the Seventh Pay Commission, announcement of which was done on 4 February 2014.[19] On 29 June 2016, Government accepted the recommendation of 7th Pay Commission Report with meager increase in salary of 14% after six month of intense evaluation and successive discussions. The Finance Minister of India claimed it historical increase of salaries due to little knowledge of Sixth Pay Commission.[citation needed] | END ID: 542

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 543 | TITLE: Lake Poets | CONTENT: There was a certain amount of additional irony involved in the 'School's' perception by readers, who were inspired, upon reading the poetry, to visit the area, thus helping to destroy, in the mind of Wordsworth at least, the very thing that made the Lakes special (although he himself ended up writing one of the best guides to the region). In addition, many of the first and second generation practitioners of Romantic poetry had a complex and not entirely easy relationship with the Lakes (apart from Wordsworth). "For the most part other Romantic poets either struggle with a Lake Poet identity or come to define themselves against what the Lakes seem to offer in poetic terms." [3] | END ID: 543

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 544 | TITLE: List of cities in Uttar Pradesh by population | CONTENT: Lucknow, Capital of Uttar Pradesh | END ID: 544

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 545 | TITLE: Novak Djokovic | CONTENT: After winning his first Master Series title, Djokovic returned to Serbia to help his country enter the Davis Cup World Group[59] in a match against Georgia. Djokovic won a point by defeating Georgia's George Chanturia.[60] Later, he played in the Monte Carlo Masters, where he was defeated by David Ferrer in the third round, and at the Estoril Open, where he defeated Richard Gasquet in the final.[61] Djokovic then reached the quarterfinals of both the Internazionali d'Italia in Rome, where he lost to Nadal, and the Hamburg Masters, where he was defeated by Carlos MoyÃ . At the French Open, Djokovic reached his first major semi-final, losing to eventual champion Nadal.[62] | END ID: 545

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 546 | TITLE: Foreign relations of India | CONTENT: Morocco has an embassy in New Delhi. It also has an Honorary Consul based in Mumbai. India operates an embassy in Rabat. Both nations are part of the Non-Aligned Movement.[425] | END ID: 546

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 547 | TITLE: Competition | CONTENT: Finally, competition also exists between governments. Each country or nationality struggles for world dominance, power, or military strength. For example, the United States competed against the Soviet Union in the Cold War for world power, and the two also struggled over the different types of government (in these cases representative democracy and communism). The result of this type of competition often leads to worldwide tensions, and may sometimes erupt into warfare. | END ID: 547

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 548 | TITLE: Pubic symphysis | CONTENT: Symphysiolysis is separation or slipping of the symphysis. It has been estimated to occur in 0.2% of pregnancies.[4] | END ID: 548

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 549 | TITLE: Nancy Cartwright | CONTENT: In 2007, Cartwright was in a relationship with contractor Stephen Brackett.[80] He was a fellow member of Scientology.[81] The couple had planned to get married in early 2008.[17][81] Brackett died in May 2009, after he "apparently leaped" off the Bixby Creek Bridge in Big Sur, California.[82] | END ID: 549

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 550 | TITLE: Where Are Ü Now | CONTENT: "Where Are Ü Now" is a song produced by American EDM artists Skrillex and Diplo under their collaborative effort Jack Ü, with vocals from Canadian singer Justin Bieber. The song was released as the second single from the duo's debut studio album, Skrillex and Diplo Present Jack Ü (2015), on their respective labels OWSLA and Mad Decent, and is also included on Bieber's fourth studio album Purpose (2015). It was released simultaneously with the album on February 27, 2015, later sent to mainstream radio on April 21, 2015. | END ID: 550

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 551 | TITLE: United States housing bubble | CONTENT: By July 2008, year-to-date prices had declined in 24 of 25 U.S. metropolitan areas, with California and the southwest experiencing the greatest price falls. According to the reports, only Milwaukee had seen an increase in house prices after July 2007.[69] | END ID: 551

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 552 | TITLE: Marilyn vos Savant | CONTENT: She expounded on her reasoning in a second follow-up and called on school teachers to show the problem to classes. In her final column on the problem, she gave the results of more than 1,000 school experiments. Most respondents now agree with her original solution, with half of the published letters declaring their authors had changed their minds.[21] | END ID: 552

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 553 | TITLE: The Messenger (Zusak novel) | CONTENT: In 2015 the novel was adapted for stage by Xavier Hazard and Archie Stapleton and performed by the Redfoot Youth Theatre Company in Perth, Western Australia.[4] | END ID: 553

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 554 | TITLE: Cory | CONTENT: Alternative spellings for Cory are Corey, Coire, Corie, Corrie, Curry (surname), Correy, Kory, Khouri, and Kori. | END ID: 554

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 555 | TITLE: Deuterocanonical books | CONTENT: Thus Jerome acknowledged the principle by which the canon would be settled â€“ the judgment of the Church (at least the local churches in this case) rather than his own judgment or the judgment of Jews; though concerning translation of Daniel to Greek, he wondered why one should use the version of a translator whom he regarded as heretic and judaizer (Theodotion).[79] | END ID: 555

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 556 | TITLE: Constitution of the People's Republic of China | CONTENT: The 1982 document reflects Deng Xiaoping's determination to lay a lasting institutional foundation for domestic stability and modernization. The new State Constitution provides a legal basis for the broad changes in China's social and economic institutions and significantly revises government structure. | END ID: 556

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 557 | TITLE: List of Game of Thrones characters | CONTENT: Tyrion Lannister (season 1â€“present) portrayed by Peter Dinklage. Nicknamed "The Imp" or "Halfman", Tyrion Lannister of House Lannister is the younger brother of Cersei and Jaime Lannister. He is a dwarf; and his mother died during his birth, for which his father, Tywin Lannister, blames him. While not physically powerful, Tyrion has a cunning mind and often uses to his advantage the fact that others constantly underestimate him. In Season Three, Tyrion is given the job of Master of Coin, a role that he is unprepared and inexperienced for. Tyrion is commanded by his father to marry Sansa Stark; however, on the wedding night, Tyrion refuses to consummate his marriage and instead lets Sansa sleep alone, promising not to touch her unless she wanted him to. The death of her brother Robb, in which Tyrion took no part, causes a further rift between the couple and between Tyrion and his father, who he claims can't distinguish between his interests and his often-praised ideal of devotion to family. Tywin bitterly claims that he had wanted to drown Tyrion upon birth, but stayed himself for the sake of duty. In season 4, Tyrion welcomes Prince Oberyn Martell of Dorne to King's Landing for Joffrey's wedding to Margaery Tyrell, though Oberyn implies to Tyrion that his true purpose is to seek vengeance for his sister, who was murdered by Ser Gregor Clegane on Tywin's orders. When Joffrey is fatally poisoned, Tyrion is framed and arrested. Tyrion, however, implies that Cersei knows of his innocence and just wants him dead. At Tyrion's trial, he demands a trial by combat. Tyrion is approached by Oberyn, who volunteers to be his champion in order to fight Cersei's champion, Ser Gregor Clegane. When Oberyn loses the fight and is killed, Tyrion is sentenced to death. Jaime, however, frees Tyrion and arranges for him to escape King's Landing. Tyrion goes to confront Tywin in his chambers but finds Shae, who testified against him and is now Tywin's lover. After a brief struggle, Tyrion strangles Shae to death, and Tyrion shoots Tywin to death with Joffrey's crossbow. Tyrion is then placed in a crate and smuggled off to Essos with help from Varys. They arrive in Pentos, where Varys manages to convince Tyrion to travel with him to Meereen and aid Daenerys Targaryen in retaking the Iron Throne. Tyrion is bound and gagged by Jorah Mormont, who says that he will take him to Daenerys. Daenerys takes them both to her home in the Great Pyramid of Meereen and asks Tyrion why he is here. Tyrion tells her everything, including Varys' plan. He initially counsels her to stay in Meereen, but Daenerys makes it plain to him that her eyes are still on Westeros. Tyrion tells Daenerys how hard it will be to win the love of both the common people and the nobles. He later joins her at the opening celebrations of Daznak's Pit, where Jorah unexpectedly reappears to defeat every other foe on the arena. As the Sons of the Harpy attack, Tyrion manages to survive by fleeing to the midst of the arena, where they are rescued by Drogon, while Daenerys is spirited away on his back. In season 6, Tyrion struggles to maintain peace in Meereen, particularly when the Sons of the Harpy burn the entire Meereenese Navy, stranding them in Slaver's Bay. Following an alliance with Theon and Yara Greyjoy, Tyrion advises Daenerys to break up with Daario so that she may pursue a marriage alliance in Westeros. In gratitude for Tyrion's loyalty, Daenerys names him her official Hand, and he accompanies her back to Westeros. In season 7 Tyrion's advice to Daenerys often conflicts with her views, and he has mixed success in directing her actions. He sneaks into King's Landing and meets with Jamie to set up a meeting between Cersei and Daenerys, which eventually happens in the Dragonpit outside the city. At the meeting a captured wight is produced to convince Cersei that the worst threat is from beyond the Wall. When Cersei is not convinced, Tyrion risks execution to meet with his sister to try again to persuade her. He is unexpectedly allowed to leave Cersei's presence alive, and thinks he has secured an agreement from her to help with the fight against the Wight Walkers. | END ID: 557

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 558 | TITLE: List of Toy Story characters | CONTENT: Angel Kitty is a Christmas ornament that only appears in Toy Story That Time Forgot. A running gag in the film is Angel Kitty giving a moral about Christmas much to other toys' (mostly Trixie) dismay and joy. She is mostly seen with a trumpet giving morals. She was last seen in Toy Story That Time Forgot giving one last moral and "vanishes". | END ID: 558

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 559 | TITLE: Manhattan Project | CONTENT: The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada. From 1942 to 1946, the project was under the direction of Major General Leslie Groves of the U.S. Army Corps of Engineers. Nuclear physicist Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual bombs. The Army component of the project was designated the Manhattan District; "Manhattan" gradually superseded the official codename, Development of Substitute Materials, for the entire project. Along the way, the project absorbed its earlier British counterpart, Tube Alloys. The Manhattan Project began modestly in 1939, but grew to employ more than 130,000 people and cost nearly US $2 billion (about $22 billion in 2016[1] dollars). Over 90% of the cost was for building factories and to produce fissile material, with less than 10% for development and production of the weapons. Research and production took place at more than 30 sites across the United States, the United Kingdom, and Canada. | END ID: 559

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 560 | TITLE: Kyon Ki | CONTENT: Kyon Ki released on 2 November 2005 to coincide with the festival of Diwali in India. It performed poorly at the box office and grossed over ₹231 million. Another Priyadarshan-directed film was released on the same day, the comedy Garam Masala which was commercially successful at the box office, grossing over ₹546 million.[2][3][4] | END ID: 560

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 561 | TITLE: Alchemy | CONTENT: Though most of these appointments were legitimate, the trend of pseudo-alchemical fraud continued through the Renaissance. BetrÃ¼ger would use sleight of hand, or claims of secret knowledge to make money or secure patronage. Legitimate mystical and medical alchemists such as Michael Maier and Heinrich Khunrath wrote about fraudulent transmutations, distinguishing themselves from the con artists.[74] False alchemists were sometimes prosecuted for fraud. | END ID: 561

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 562 | TITLE: Fútbol de Primera (radio network) | CONTENT: San Francisco-San Jose, CA | END ID: 562

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 563 | TITLE: Wonder Woman | CONTENT: The New 52 | END ID: 563

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 564 | TITLE: Memphis, Tennessee | CONTENT: Memphis also has non-commercial visual arts organizations and spaces, including local painter Pinkney Herbert's Marshall Arts gallery, on Marshall Avenue near Sun Studios, another arts neighborhood characterized by affordable rent.[92] | END ID: 564

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 565 | TITLE: An Lushan Rebellion | CONTENT: The An Lushan Rebellion signaled a period of disorder spanning the reigns of three Tang dynasty emperors, beginning during the final (Tianbao era) period of the reign of Xuanzong (8 September 712-12 August 756), continuing through the reign of Suzong (12 August 756-16 May 762) and ending during the reign of Daizong (18 May 762-23 May 779), as well as spanning the four imperial claimants of the failed Da Yan dynasty. | END ID: 565

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 566 | TITLE: Racing Post | CONTENT: Star writers and columnists include: | END ID: 566

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 567 | TITLE: Hydroelectricity | CONTENT: Construction suspended 14 days by court order Aug 2012[55] | END ID: 567

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 568 | TITLE: Kevin Malone | CONTENT: In "Health Care", it was revealed that he suffers from anal fissures.[7] He is the resident gambler, partaking or initiating various bets throughout the office with other co-workers, and shows a fairly sharp talent for monitoring games fairly. In "Casino Night", he reveals that he won a World Series of Poker bracelet for No-Limit Deuce-Seven Triple Draw in 2002, but he nonetheless suffers defeat at the hands of an unwitting Phyllis Lapin-Vance. Kevin also enjoys cooking. As seen in the "Kevin Cooks Stuff in The Office" short, Kevin is currently brewing beer in the cabinet beside his desk, much to the disgust of his coworker Oscar Martinez. Kevin states that, "rules say I can't bring beer into the office, but they don't say anything about making beer in the office." He also has recipes for office-made quesadillas, creme brulee, and mashed potatoes and once a year brings in a large pot of his "famous" Chili for the office. Kevin and Oscar's personalities complement each other, with Oscar being the most intelligent Dunder-Mifflin employee and Kevin being arguably the least, and they are shown to be close friends. The two often do a fist bump when they have a minor achievement such as playing office games. | END ID: 568

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 569 | TITLE: I Can See Your Voice | CONTENT: Skilled Vocalist: | END ID: 569

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 570 | TITLE: Alcohol by volume | CONTENT: During the production of wine and beer, yeast is added to a sugary solution. During fermentation, the yeasts consume the sugars and produce alcohol. The density of sugar in water is greater than the density of alcohol in water. A hydrometer is used to measure the change in specific gravity (SG) of the solution before and after fermentation. The volume of alcohol in the solution can then be estimated. There are a number of empirical formulae which brewers and winemakers use to estimate the alcohol content of the liquor made. | END ID: 570

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 571 | TITLE: The Nut Job | CONTENT: In a post-credits scene, Precious chases Mole to get the bone he is holding that she wants and he drives her away with the dog whistle. | END ID: 571

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 572 | TITLE: Tropic hormone | CONTENT: Tropic hormones are hormones that have other endocrine glands as their target. Most tropic hormones are produced and secreted by the anterior pituitary.[1] The hypothalamus secretes tropic hormones that target the anterior pituitary, and the thyroid gland secretes thyroxine, which targets the hypothalamus and therefore can be considered a tropic hormone.[2] | END ID: 572

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 573 | TITLE: Canada's role in the War in Afghanistan | CONTENT: In February 2008, the Van Doos contingent was replaced by force centred on a PPCLI battle group. Also in February 2008, Canadian Major-General Marc Lessard took command of Regional Command South for a nine-month period. | END ID: 573

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 574 | TITLE: Police community support officer | CONTENT: Male PCSOs wear flat caps rather than custodian helmets, which are worn by male police constables. The Metropolitan Police Authority noted in 2004 that the hats worn by male PCSOs were not rigid and 'may therefore not offer adequate protection'.[36][37] Female PCSOs wear bowler hats contain foam padding as protection. | END ID: 574

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 575 | TITLE: Cornwall Council | CONTENT: Among the services provided by the council is a public library service which consists of a main library in Truro and smaller libraries in towns and some villages throughout Cornwall. There are also the following special libraries: Cornwall Learning Library, Cornish Studies Library, the Education Library Service, and the Performing Arts Library, as well as a mobile library service based at Threemilestone.[17] | END ID: 575

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 576 | TITLE: Artificial heart | CONTENT: In 1981, William DeVries submitted a request to the FDA for permission to implant the Jarvik 7 into a human being. On December 2, 1982, Kolff implanted the Jarvik 7 artificial heart into Barney Clark, a dentist from Seattle who was suffering from severe congestive heart failure. Clark lived for 112 days tethered to an external pneumatic compressor, a device weighing some 400 pounds (180Â kg), but during that time he suffered prolonged periods of confusion and a number of instances of bleeding, and asked several times to be allowed to die.[11] | END ID: 576

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 577 | TITLE: Scion of Ikshvaku | CONTENT: Tripathi wrote the story both from a critical point of view about Rama as well as a devotee of him, adding that "part of our traditions is also to learn from the stories of our gods".[10] He also confirmed a part of the plot, where Ravana wins a war in the story and enforces a trade deal which results in the Sapta Sindhu area to give economic privileges to Lanka.[15] Like the characterization of women in the Shiva trilogy, Tripathi had strong female perspective in Scion of Ikshvaku, including portraying Manthara as a businesswomen. This was a deviation from the original story, where she was a servant. Other concepts explored included the rise and fall of masculine and feminine centric civilisations, as well as using scientific evidence for making the character of Hanuman from the epic as a Naga, a concept introduced in the Shiva trilogy.[16] | END ID: 577

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 578 | TITLE: Normandy | CONTENT: And new economic activity stimulated the coasts: seaside tourism. The 19th century marks the birth of the first beach resorts. | END ID: 578

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 579 | TITLE: List of The X Factor finalists (U.S. season 2) | CONTENT: The "Young Adults" category was mentored by Demi Lovato. The six candidates were Jennel Garcia, Willie Jones, Nick Youngerman, Paige Thomas, Jillian Jensen and CeCe Frey. Lovato chose the following: | END ID: 579

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 580 | TITLE: Stock (firearms) | CONTENT: Some modern buttstock has a movable comb piece called a cheek rest or cheek rise, which offers adjustable comb height that tailors to the shooter's ergonomic preference. | END ID: 580

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 581 | TITLE: List of pharaohs | CONTENT: Note that the dates given are approximate. The list of pharaohs presented below is based on the conventional chronology of Ancient Egypt, mostly based on the Digital Egypt for Universities database developed by the Petrie Museum of Egyptian Archaeology, but alternative dates taken from other authorities may be indicated separately. | END ID: 581

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 582 | TITLE: Property insurance | CONTENT: In May 2007 New York Governor Eliot Spitzer announced more than $4.5 billion would be made available to rebuild the 16-acre (65,000 m2) WTC complex as part of a major insurance claims settlement.[9] | END ID: 582

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 583 | TITLE: WWE Raw | CONTENT: On September 24, 2012, Hulu Plus signed a multi-year deal with WWE to stream all of the companyâ€™s TV shows and some of its web series which includes Raw. Episodes of Raw are available for viewing the following day and only a condensed 90 minute version is available, not the full version as shown the previous night on the USA Network.[58] | END ID: 583

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 584 | TITLE: Xylem | CONTENT: While wider tracheids with robust walls make it possible to achieve higher water transport pressures, this increases the problem of cavitation.[27] Cavitation occurs when a bubble of air forms within a vessel, breaking the bonds between chains of water molecules and preventing them from pulling more water up with their cohesive tension. A tracheid, once cavitated, cannot have its embolism removed and return to service (except in a few advanced angiosperms[verification needed] which have developed a mechanism of doing so). Therefore, it is well worth plants' while to avoid cavitation occurring. For this reason, pits in tracheid walls have very small diameters, to prevent air entering and allowing bubbles to nucleate. Freeze-thaw cycles are a major cause of cavitation. Damage to a tracheid's wall almost inevitably leads to air leaking in and cavitation, hence the importance of many tracheids working in parallel.[27] | END ID: 584

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 585 | TITLE: Lena Headey | CONTENT: She joined again with Ethan Hawke to co-star in The Purge, a 'micro-budget' horror film, which opened on 7 June 2013 at number one position in the United States, grossing over $US36 million over the weekend.[35] | END ID: 585

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 586 | TITLE: Glorfindel | CONTENT: Glorfindel is also playable in the older Middle-earth Collectible Card Game. Here he is one of the most powerful characters outside the circle of the Wizards and Haven-elves (Elrond, Galadriel and Círdan). | END ID: 586

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 587 | TITLE: Spectral line | CONTENT: Without qualification, "spectral lines" generally implies that one is talking about lines with wavelengths which fall into the range of the visible spectrum. However, there are also many spectral lines which show up at wavelengths outside this range. At the much shorter wavelengths of x-rays, these are known as characteristic X-rays. Other frequencies have atomic spectral lines as well, such as the Lyman series, which falls in the ultraviolet range. | END ID: 587

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 588 | TITLE: Denny's | CONTENT: In 1997, six Asian-American students from Syracuse University visited a local Denny’s restaurant late at night. They waited for more than half an hour as white patrons were regularly served, seated, and offered more helpings. They complained to management and to their server but were forced to leave the establishment by two security guards called by Denny’s management. Then, according to the students, a group of white men came out of Denny's, attacked them[28] and shouted racial epithets. Several of the students were beaten into unconsciousness.[29][30] | END ID: 588

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 589 | TITLE: Python (programming language) | CONTENT: CPython is the reference implementation of Python. It is written in C, meeting the C89 standard with several select C99 features.[97] It compiles Python programs into an intermediate bytecode[98] which is then executed by its virtual machine.[99] CPython is distributed with a large standard library written in a mixture of C and native Python. It is available for many platforms, including Windows and most modern Unix-like systems. Platform portability was one of its earliest priorities.[100] | END ID: 589

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 590 | TITLE: What Did I Do to Deserve This, My Lord? | CONTENT: The game was released in North America exclusively as a download game on the PlayStation Store, under the title Holy Invasion of Privacy, Badman! What Did I Do To Deserve This?.[1] However, on February 9, 2010, NIS America revealed it would be changing the game's name to avoid conflict with the Batman franchise. The game was re-released on April 22, 2010 on the PlayStation Network after it was removed to make the changes, while its sequel, What Did I Do To Deserve This, My Lord? 2, had been delayed to May 4, 2010.[3] | END ID: 590

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 591 | TITLE: Breakfast television | CONTENT: A few of the major Spanish language broadcast networks also produce morning shows, which are often more festive in format. ¡Despierta América! (Wake Up America!) is the longest-running Spanish language morning program on U.S. network television having aired on Univision since April 1997; Telemundo made several failed attempts at hard news and traditional morning shows during the 1990s and 2000s before it finally experienced success with Un Nuevo Día (A New Day), which launched in 2008 under the title ¡Levántate! (Get Up), and became a formidable competitor to its longer established rival following a 2011 format retooling. | END ID: 591

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 592 | TITLE: E! | CONTENT: In recent years, the network has become well known for its reality television programs. Its most popular series as of 2011 is Keeping Up with the Kardashians, which has spawned three spin-offs (Kourtney and Khloé Take Miami, Kourtney and Kim Take New York, and Khloe and Lamar). Other original programming airing on the network included weekly version of Fashion Police (which continues as post-awards ceremony specials). | END ID: 592

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 593 | TITLE: Pluto (mythology) | CONTENT: Plūtō ([ˈpluːtoː]; genitive Plūtōnis) is the Latinized form of the Greek Plouton. Pluto's Roman equivalent is Dis Pater, whose name is most often taken to mean "Rich Father" and is perhaps a direct translation of Plouton. Pluto was also identified with the obscure Roman Orcus, like Hades the name of both a god of the underworld and the underworld as a place. The borrowed Greek name Pluto is sometimes used for the ruler of the dead in Latin literature, leading some mythology handbooks to assert misleadingly that Pluto was the Roman counterpart of Hades.[4] Pluto (Pluton in French and German, Plutone in Italian) becomes the most common name for the classical ruler of the underworld in subsequent Western literature and other art forms. | END ID: 593

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 594 | TITLE: History of money | CONTENT: When the inhabitants of one country became more dependent on those of another, and they imported what they needed, and exported what they had too much of, money necessarily came into use.[14] | END ID: 594

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 595 | TITLE: Stephen Harper | CONTENT: Harper was the only Reform MP to support the creation of the Canadian Firearms Registry at second reading in 1995, although he later voted against it at third reading stage. He said at the time that he initially voted for the registry because of a poll showing that most of his constituents supported it, and added that he changed his vote when a second poll showed the opposite result. It was reported in April 1995, that some Progressive Conservatives opposed to Jean Charest's leadership wanted to remove both Charest and Manning, and unite the Reform and Progressive Conservative parties under Harper's leadership.[39] | END ID: 595

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 596 | TITLE: Binary star | CONTENT: By the modern definition, the term binary star is generally restricted to pairs of stars which revolve around a common center of mass. Binary stars which can be resolved with a telescope or interferometric methods are known as visual binaries.[5][6] For most of the known visual binary stars one whole revolution has not been observed yet, they are observed to have travelled along a curved path or a partial arc.[7] | END ID: 596

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 597 | TITLE: Goku | CONTENT: Dragon Ball GT chief character designer Katsuyoshi Nakatsuru said he agonized over designing Goku's Super Saiyan 4 appearance, which was the idea of the show's producers, questioning whether it was necessary to go further with the transformations. Because Super Saiyan 4 is brought about while in a Saiyan's Ōzaru (大猿, lit. "Great Ape") form, he made the hair more "wild" and covered Goku's body in red fur. There was only a single final draft of the character, although Nakatsuru did consider making the hair blonde, he ended up choosing black as it provides more contrast with the red fur.[16] | END ID: 597

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 598 | TITLE: New Zealand Warriors | CONTENT: 2001â€“2002 | END ID: 598

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 599 | TITLE: Three-point field goal | CONTENT: The three-point line was first tested at the collegiate level in a 1945 NCAA game between Columbia and Fordham but it was not kept as a rule. At the direction of Abe Saperstein, the American Basketball League became the first basketball league to institute the rule in 1961. Its three-point line was a radius of 25 feet (7.62 m) from the baskets, except along the sides.[2] The Eastern Professional Basketball League followed in its 1963–64 season. | END ID: 599

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 600 | TITLE: Colonial charters in the Thirteen Colonies | CONTENT: The Virginia and Massachusetts charters were given to business corporations. Regular meetings of company officers and stockholders were the only governmental institutions required. The Virginia charter, issued in 1606, and revised in 1609 and 1612, was revoked upon bankruptcy of the sponsoring and organizing Virginia Company of London in 1624. The second Colonial Charter was granted to Massachusetts Bay in 1629, settling at Boston and Salem, a decade after the first "New Englanders" at Plymouth Colony further south towards Cape Cod. In 1684, the Chancery Court in England voided the charter and changed it to a royal colony. Charles II placed Massachusetts under the authority of the unified Dominion of New England in 1685. After William III came to the throne, he issued Massachusetts Bay a new liberal charter in 1691. | END ID: 600

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 601 | TITLE: Communications satellite | CONTENT: Wireless communication uses electromagnetic waves to carry signals. These waves require line-of-sight, and are thus obstructed by the curvature of the Earth. The purpose of communications satellites is to relay the signal around the curve of the Earth allowing communication between widely separated points.[2] Communications satellites use a wide range of radio and microwave frequencies. To avoid signal interference, international organizations have regulations for which frequency ranges or "bands" certain organizations are allowed to use. This allocation of bands minimizes the risk of signal interference.[3] | END ID: 601

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 602 | TITLE: Crown Royal | CONTENT: The brand was a primary sponsor of the Washington International Horse Show for several years in the 1990s and since 1995 has sponsored the Crown Royal American Turf Stakes, a Thoroughbred horse race run annually at Churchill Downs. | END ID: 602

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 603 | TITLE: Flow banding | CONTENT: Flow banding is caused by friction of the viscous magma that is in contact with a solid rock interface, usually the wall rock to an intrusive chamber or, if the magma is erupted, the surface of the earth across which the lava is flowing. | END ID: 603

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 604 | TITLE: Race and ethnicity in the United States | CONTENT: The United States Census Bureau is presently finalizing the ethnic classification of MENA populations. In 2012, prompted in part by post-9/11 discrimination, the American-Arab Anti-Discrimination Committee petitioned the Department of Commerce's Minority Business Development Agency to designate the MENA populations as a minority/disadvantaged community.[77] Following consultations with MENA organizations, the US Census Bureau announced in 2014 that it would establish a new MENA ethnic category for populations from the Middle East, North Africa and the Arab world, separate from the "white" classification that these populations had previously sought in 1909. The expert groups, including some Jewish organizations, felt that the earlier "white" designation no longer accurately represents MENA identity, so they successfully lobbied for a distinct categorization.[16][78] This process does not currently include ethnoreligious groups such as Jews or Sikhs, as the Bureau only tabulates these groups as followers of religions rather than members of ethnic groups.[79] | END ID: 604

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 605 | TITLE: Pradhan Mantri Gramin Awaas Yojana | CONTENT: Under the scheme,[13] eligible people will get a financial assistance from government amounting to ₹1.2 lakh (US$1,800) for constructing their houses in rural areas and an amount of ₹12,000 (US$180) for constructing toilets. In addition, they can also borrow an amount of ₹70,000 (US$1,100).[8] After current provision of PMGAY people should apply online.[14] | END ID: 605

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 606 | TITLE: Cancer staging | CONTENT: In situ neoplasia identified microscopically during the diagnostic workup may be used to assign the pathological stage pTis if the patient had a surgical resection and no residual tumor was identified. | END ID: 606

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 607 | TITLE: Heroes and Villains (Only Fools and Horses) | CONTENT: "Heroes and Villains" is an episode of the BBC sitcom, Only Fools and Horses, first screened on 25 December 1996 as the first part of the 1996 Christmas trilogy and the thirteenth Christmas special. It attracted a UK television audience of 21.3 million, at the time a record for the show. In the episode, Del and Rodney are invited to a fancy dress party. They arrive dressed as Batman and Robin. | END ID: 607

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 608 | TITLE: Human hair color | CONTENT: Blond hair can have almost any proportion of pheomelanin and eumelanin, but has only small amounts of both. More pheomelanin creates a more golden or strawberry blond color, and more eumelanin creates an ash or sandy blond color. Many children born with blond hair develop darker hair as they age, with the majority of natural blonds developing a hair color of a dark blond hue by the time they reach middle age. Pregnancy hormones hasten this process. Natural light blond hair is rare in adulthood, with claims of the world's population ranging from 2% naturally blond[3][self-published source] to 16% in the US.[4] Blond hair is most commonly found in Northern and Western Europeans and their descendants but can be found spread around most of Europe. Studies in 2012 showed that naturally blond hair of Melanesians is caused by a recessive mutation in tyrosinase-related protein 1 (TYRP1). In the Solomon Islands, 26% of the population carry the gene; however, it is absent outside of Oceania.[5] | END ID: 608

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 609 | TITLE: Qing dynasty | CONTENT: The dynasty was founded by the Jurchen Aisin Gioro clan in Manchuria. In the late sixteenth century, Nurhaci, originally a Ming vassal, began organizing "Banners", military-social units that included Jurchen, Han Chinese, and Mongol elements. Nurhaci formed the Jurchen clans into a unified entity, which he renamed as the Manchus. By 1636, his son Hong Taiji began driving Ming forces out of Liaodong and declared a new dynasty, the Qing. In 1644, peasant rebels led by Li Zicheng conquered the Ming capital, Beijing. Rather than serve them, Ming general Wu Sangui made an alliance with the Manchus and opened the Shanhai Pass to the Banner Armies led by the regent Prince Dorgon, who defeated the rebels and seized the capital. Resistance from the Southern Ming and the Revolt of the Three Feudatories led by Wu Sangui extended the conquest of China proper for nearly four decades and was not completed until 1683 under the Kangxi Emperor (r. 1661â€“1722). The Ten Great Campaigns of the Qianlong Emperor from the 1750s to the 1790s extended Qing control into Inner Asia. The early rulers maintained their Manchu ways, and while their title was Emperor, they used "Bogd khaan" to the Mongols and they were patrons of Tibetan Buddhism. They governed using Confucian styles and institutions of bureaucratic government and retained the imperial examinations to recruit Han Chinese to work under or in parallel with Manchus. They also adapted the ideals of the tributary system in dealing with neighboring territories. | END ID: 609

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 610 | TITLE: United States Constitution | CONTENT: Salmon P. Chase was a Lincoln appointee, serving as Chief Justice from 1864 to 1873. His career encompassed service as a U.S. Senator and Governor of Ohio. He coined the slogan, "Free soil, free Labor, free men." One of Lincoln's "team of rivals", he was appointed Secretary of Treasury during the Civil War, issuing "greenbacks". To appease radical Republicans, Lincoln appointed him to replace Chief Justice Roger B. Taney of Dred Scott case fame. | END ID: 610

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 611 | TITLE: Clement Attlee | CONTENT: Attlee died peacefully in his sleep of pneumonia, at the age of 84 at Westminster Hospital on 8 October 1967.[182] 2,000 people attended his funeral in November, including the then-Prime Minister Harold Wilson and the Duke of Kent, representing the Queen. He was cremated and his ashes were buried at Westminster Abbey.[189][190][191] | END ID: 611

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 612 | TITLE: Expansion of Major League Soccer | CONTENT: After a two-year hiatus, the San Jose Earthquakes were reactivated in 2007 and resumed play in MLS in 2008.[23] | END ID: 612

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 613 | TITLE: Felicia Day | CONTENT: On January 3, 2017, Day announced on social media that she was pregnant and expecting a baby girl in a few weeks.[47][48] She announced on January 30, 2017, that her daughter, Calliope Maeve, had been born.[49] | END ID: 613

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 614 | TITLE: Oz (Buffy the Vampire Slayer) | CONTENT: He notices Willow in her Eskimo costume at a dance at The Bronze, and seems to be interested in her at first sight; but does not meet her directly until several episodes later. They have several dates, on one of which he witnesses a vampire being dusted by Buffy for the first time, and is unsurprised upon learning vampires exist and merely remarks that "it explains a lot". After this he becomes a member of the Scooby Gang, helping with research and fighting. | END ID: 614

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 615 | TITLE: Names of China | CONTENT: Chinese names for China, aside from Zhongguo, include Zhonghua (中華/中华), Huaxia (華夏/华夏), Shenzhou (神州) and Jiuzhou (九州). Han (漢/汉) and Tang (唐) are common names given for the Chinese ethnicity. The People's Republic of China (Zhōnghuá Rénmín Gònghéguó) and Republic of China (Zhōnghuá Mínguó) are the official names for the two contemporary sovereign states currently claiming sovereignty over the traditional area of China. "Mainland China" is used to refer to areas under the jurisdiction by the PRC usually excluding Hong Kong and Macau. | END ID: 615

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 616 | TITLE: Chongqing University | CONTENT: Faculty of Information Science and Technology | END ID: 616

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 617 | TITLE: List of national parks of India | CONTENT: Further federal legislation strengthening protections for wildlife was introduced in the 1980s. As of July 2017, there were 103 national parks encompassing an area of 40,500 km2 (15,600 sq mi), comprising 1.23% of India's total surface area.[1] | END ID: 617

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 618 | TITLE: History of the Metropolitan Police Service | CONTENT: In 1981, a report issued by Lord Scarman stated that the Metropolitan Police were having problems regarding racial discrimination.[35] The issue arose again in the 1999 Macpherson Report, which stated that institutional racism existed in the force.[36] | END ID: 618

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 619 | TITLE: United States presidential election, 2020 | CONTENT: Trump vs. Booker | END ID: 619

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 620 | TITLE: Pernell Roberts | CONTENT: Roberts played Ben Cartwright's urbane eldest son Adam, in the Western television series Bonanza. Unlike his brothers, Adam was a university-educated architectural engineer. | END ID: 620

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 621 | TITLE: Tracie Young | CONTENT: Tracie Young (often just billed as Tracie; born 1965) is a former English pop singer in the 1980s. She achieved success after becoming a protégée of Paul Weller. | END ID: 621

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 622 | TITLE: Catholic laity | CONTENT: An interval, determined by the Holy See or the conferences of bishops, shall be observed between the conferring of the ministries of reader and acolyte whenever more than one ministry is conferred on the same person."[8] | END ID: 622

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 623 | TITLE: Ship | CONTENT: For ships with independent propulsion systems for each side, such as manual oars or some paddles,[69] steering systems may not be necessary. In most designs, such as boats propelled by engines or sails, a steering system becomes necessary. The most common is a rudder, a submerged plane located at the rear of the hull. Rudders are rotated to generate a lateral force which turns the boat. Rudders can be rotated by a tiller, manual wheels, or electro-hydraulic systems. Autopilot systems combine mechanical rudders with navigation systems. Ducted propellers are sometimes used for steering. | END ID: 623

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 624 | TITLE: First Epistle of John | CONTENT: The First Epistle of John, often referred to as First John and written 1 John, is the first of the Johannine epistles of the New Testament, and the fourth of the catholic epistles. It is attributed to John the Evangelist, traditionally thought to be the author of the Gospel of John and the other two Johannine epistles. This epistle was probably written in Ephesus in AD 95–110.[1] The work was written to counter docetism, which is the belief that Jesus did not come "in the flesh", but only as a spirit. It also defined how Christians are to discern true teachers: by their ethics, their proclamation of Jesus in the flesh, and by their love.[1] | END ID: 624

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 625 | TITLE: German invasion of Belgium | CONTENT: German attacks south of the Menin road took small areas but Messines ridge had been consolidated by the British garrison and was not captured. By 1 November, the BEF was close to exhaustion and 75 of 84 infantry battalions had fewer than 300 men left; ​1⁄3 of their establishment. The French XIV Corps was moved north from the Tenth Army and the French IX Corps attacked southwards towards Becelaere, which relieved the pressure on both British flanks. German attacks began to diminish on 3 November, by when Armeegruppe von Fabeck had lost 17,250 casualties. A French offensive was planned for 6 November towards Langemarck and Messines, to widen the Ypres salient but German attacks began again on 5 November in the same area until 8 November, then again on 10–11 November. The main attack on 10 November was made by the 4th Army between Langemarck and Dixmude, in which Dixmude was lost by the Franco-Belgian garrison. Next day to the south, the British were subjected to an unprecedented bombardment between Messines and Polygon Wood and then an attack by Prussian Guard, which broke into British positions along the Menin road, before being forced back by counter-attacks.[50] From mid-October to early November the German Fourth Army lost 52,000 and the Sixth Army lost 28,000 casualties.[51] | END ID: 625

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 626 | TITLE: Chris Elliott | CONTENT: In 1986 Elliott starred in the Cinemax special FDR: A One Man Show, a spoof comedy about the life and times of the president. He looked and sounded nothing like the man; he portrayed events from Roosevelt's life that never happened, such as a Japanese bombing of the White House, and his crossing the Potomac in a rowboat. By the end of the show, he had performed Gallagher's shtick of smashing watermelons and other soft fruits on stage. | END ID: 626

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 627 | TITLE: RMS Titanic | CONTENT: RMS Titanic (/taɪˈtænɪk/) was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. The RMS Titanic was the largest ship afloat at the time it entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. The Titanic was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, her architect, died in the disaster.[2] | END ID: 627

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 628 | TITLE: Green River (Colorado River) | CONTENT: At the time of its discovery (2005), the Green River Formation was said to have the world's largest fossil fuel deposits in the form of a solid rock resource called [15] oil shale. There is estimated to be between 500 billion and 1.1 trillion barrels (80 and 175 km³) of potentially recoverable oil in the basin,[16] however; this estimated amount of recoverable oil in the form of kerogen is challenged, and in doubt, as currently there is no economically feasible technology to convert rock into a permeable oil. Kerogen is an uncooked form of hydrocarbon that nature did not convert into actual oil.[17] The cost of converting Green River oil shale into actual oil at the moment would be higher than what it could be sold for. The EROI for oil shale is very low while having a very high destructive environmental impact.[18] | END ID: 628

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 629 | TITLE: Dollar coin (United States) | CONTENT: Whatever the reason, a U.S. Mint official claimed in a November 2012 meeting that most of the 2.4 billion dollar coins minted in the previous five years were not in circulation.[8] | END ID: 629

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 630 | TITLE: Audience (TV network) | CONTENT: In the spring of 2011, DirecTV acquired the rights to the Australian series Rake.[44] | END ID: 630

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 631 | TITLE: Talk:Naturopathy/Archive 5 | CONTENT: In five Canadian provinces, fifteen US states and the District of Columbia, naturopathic doctors who are trained at an accredited school of naturopathic medicine in North America, are entitled to use the designation ND or NMD. Elsewhere, the designations "naturopath", "naturopathic doctor", and "doctor of natural medicine" are generally unprotected.[13] | END ID: 631

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 632 | TITLE: Bird migration | CONTENT: Bird migration is not limited to birds that can fly. Most species of penguin (Spheniscidae) migrate by swimming. These routes can cover over 1,000 km (620 mi). Dusky grouse Dendragapus obscurus perform altitudinal migration mostly by walking. Emus Dromaius novaehollandiae in Australia have been observed to undertake long-distance movements on foot during droughts.[12] | END ID: 632

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 633 | TITLE: Creamy layer | CONTENT: The children of persons engaged in trade, industry and professions such as a doctor, lawyer, chartered accountant, income tax consultant, financial or management consultant, dental surgeon, engineer, computer specialist, film artists and other film professional, author, playwright, sports person, sports professional, media professional or any other vocations of like status whose annual income is more than ₹ 600,000 (Rs 6 lakh) for a period of three consecutive years are also excluded. [OBC children belong to any family earning a total gross annual income (from sources other than salary and agricultural land[12][13]) of less than Rs 6 lakh for a period of three consecutive year—as the 1993 income ceiling for the creamy layer was raised from ₹ 100,000 (Rs 1 lakh, when the office memo was accepted) to Rs 6 lakh for a period of three consecutive years (in May 2013).[citation needed] Individuals belonging to the creamy layer are also excluded from being categorised as "socially and educationally backward" regardless of their social/educational backwardness.[14] | END ID: 633

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 634 | TITLE: George Washington Carver | CONTENT: Black people were not allowed at the public school in Diamond Grove. George decided to go to a school for black children 10 miles (16Â km) south in Neosho. When he reached the town, he found the school closed for the night. He slept in a nearby barn. By his own account, the next morning he met a kind woman, Mariah Watkins, from whom he wished to rent a room. When he identified himself as "Carver's George," as he had done his whole life, she replied that from now on his name was "George Carver". George liked Mariah Watkins, and her words, "You must learn all you can, then go back out into the world and give your learning back to the people", made a great impression on him.[9] | END ID: 634

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 635 | TITLE: I Wanna Be Sedated | CONTENT: "I Wanna Be Sedated" is a song by the American punk rock band the Ramones. It is one of the band's best known songs.[1] It was originally released on their fourth album, Road to Ruin, in September 1978 and was the B-side of the UK single "She's the One" released on September 21,1978.[2] The song was later released as a single in the Netherlands in 1978,[3] then in the U.S. in 1980 by RSO Records from the Times Square soundtrack album. | END ID: 635

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 636 | TITLE: Hillary Clinton email controversy | CONTENT: On September 12, 2015, Republican Senators Charles Grassley and Ron Johnson, chairmen of the Senate Judiciary and Homeland Security committees, respectively, said they would seek an independent review of the deleted emails, if they were recovered from Clinton's server, to determine if there were any government related items among those deleted.[123] | END ID: 636

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 637 | TITLE: Seashell | CONTENT: Sea shells found in the creek and backwater of the coast of west India are used as an additive to poultry feed. They are crushed and mixed with jawar maaze and dry fish.[citation needed] | END ID: 637

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 638 | TITLE: SS Atlantic Empress | CONTENT: The collision and fire claimed the lives of 26 of the Empress's crew members, and one crew member on the Captain.[3] The remaining crew from both ships were taken to Tobago for medical treatment, while the Empress's captain was transported to a hospital in Texas, having inhaled fire.[2] | END ID: 638

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 639 | TITLE: Joint Comprehensive Plan of Action | CONTENT: The IAEA, EU, Russia and China have all affirmed that Iran is respecting the limitations on its nuclear program.[554] The IAEA, the foremost authority on the matter, has repeatedly deemed Iran in compliance with the nuclear deal. The U.S. State Department has also certified that Iran is holding up its end of the bargain, and a host of experts affirmed these findings.[555] IAEA Director General Amano said that "Iran is subject to the world's most robust nuclear verification regime."[556] According to David Makovsky, a Middle East scholar at the Washington Institute for Near East Policy, Iran was not in compliance, because under the terms of the deal, Iran was supposed to reveal all of its research into nuclear weapons, and that based on evidence presented by Israeli Prime Minister Benjamin Netanyahu on April 30,2018, “it seems clear that they did not.”[557] | END ID: 639

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 640 | TITLE: Pear | CONTENT: Pears may be stored at room temperature until ripe.[17] Pears are ripe when the flesh around the stem gives to gentle pressure.[17] Ripe pears are optimally stored refrigerated, uncovered in a single layer, where they have a shelf life of 2 to 3 days.[17] | END ID: 640

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 641 | TITLE: Hap and Leonard (TV series) | CONTENT: Hap and Leonard is an American television drama series based on the characters Hap and Leonard, created by novelist Joe R. Lansdale[1] and adapted from his series of novels of the same name.[2] The series was written and developed by Nick Damici and Jim Mickle, who had previously adapted Lansdale's Cold in July and was directed by Mickle.[1] The series premiered on the American cable network SundanceTV on March 2, 2016.[1][3] So far, the series has received favorable reviews.[4][5][6] | END ID: 641

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 642 | TITLE: Jesus Christ Superstar (album) | CONTENT: All compositions written by Tim Rice (lyrics and book) and Andrew Lloyd Webber (music). | END ID: 642

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 643 | TITLE: History of film | CONTENT: The use of different camera speeds also appeared around 1900 in the films of Robert W. Paul and Hepworth. Paul shot scenes from On a Runaway Motor Car through Piccadilly Circus (1899) with the camera turning very slowly. When the film was projected at the usual 16 frames per second, the scenery appeared to be passing at great speed. Hepworth used the opposite effect in The Indian Chief and the Seidlitz Powder (1901). The Chief's movements are sped up by cranking the camera much faster than 16 frames per second. This gives what we would call a "slow motion" effect. | END ID: 643

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 644 | TITLE: Law of the United Kingdom | CONTENT: The United Kingdom has three legal systems, each of which applies to a particular geographical area.[1] English law applies in England and Wales, Northern Ireland law applies in Northern Ireland, and Scots law applies in Scotland. While these three systems diverge in the more detailed rules, there are also substantive fields of law which apply across the United Kingdom. | END ID: 644

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 645 | TITLE: Melanocyte | CONTENT: Melanocytes are melanin-producing neural crest-derived[3] cells located in the bottom layer (the stratum basale) of the skin's epidermis, the middle layer of the eye (the uvea),[4] the inner ear,[5] vaginal epithelium,[6] meninges,[7] bones,[8] and heart.[9] Melanin is a dark pigment primarily responsible for skin color. Once synthesized, melanin is contained in special organelles called melanosomes which can be transported to nearby keratinocytes to induce pigmentation. Functionally, melanin serves as protection against UV radiation. Melanocytes also have a role in the immune system. | END ID: 645

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 646 | TITLE: Squad number (association football) | CONTENT: In The Football League, the number 55 has been worn by Ade Akinbiyi, for Crystal Palace,[11] and Dominik Werling, for Barnsley,[12] | END ID: 646

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 647 | TITLE: Christopher McCandless | CONTENT: In April 1992, McCandless hitchhiked from Carthage, South Dakota, to Fairbanks, Alaska. As noted by Krakauer, McCandless was last seen alive at the head of the Stampede Trail on April 28, 1992, by a local electrician named Jim Gallien. Gallien had given McCandless a ride from Fairbanks to the start of the rugged track just outside the small town of Healy. Gallien later said he had been seriously concerned about the safety of McCandless (who introduced himself as "Alex"), after noticing McCandless' light pack, minimal equipment, meager rations, and obvious lack of experience. Gallien said he had deep doubts about "Alex"'s ability to survive the harsh and unforgiving Alaskan bush.[citation needed] | END ID: 647

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 648 | TITLE: Lloyd's of London | CONTENT: In its most recent annual report, for 2017, Lloyd's reported an underwriting loss of £3.42bn, offset by a £1.42bn non-technical profit to produce an overall pre-tax loss of £2bn, compared to an overall £2.11bn pre-tax profit in 2016. The result was driven by an increase in large claims to £4.54bn, primarily arising out of Hurricanes Harvey, Irma and Maria and wildfires in California. Gross premiums written totalled £33.59bn, which was a 12.5 per cent increase from £29.86bn in 2016, without taking exchange rate fluctuations into account. | END ID: 648

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 649 | TITLE: Burnt (film) | CONTENT: After arriving in London, Adam begins to look up his old colleagues. Jean-Luc's former maître d'hôtel, Tony (Daniel Brühl), now manages his father's hotel. Adam wants to take over the hotel's restaurant, but Tony does not trust him after his behavior in Paris. Adam soon learns that Jean-Luc died and feels deep remorse for how he hurt his mentor. Adam visits his friend Conti (Henry Goodman) at the restaurant Conti owns. He takes a liking to Conti's sous-chef Helene (Sienna Miller), but Helene finds him old-fashioned and unbearably conceited. Adam locates Michel (Omar Sy), another of his friends from Jean-Luc's, but Michel opened his own restaurant. Feeling betrayed, Adam released rats in Michel's kitchen and reported him to a health inspector, which led to the restaurant's closure. Michel forgives Adam and agrees to work for him. Adam pays a visit to Reece's, a cutting edge eatery run by Reece (Matthew Rhys), with whom he has a long-standing rivalry. Reece does not forgive him. Adam also plans to employ another Jean-Luc protege, Max (Riccardo Scamarcio), after he is released from prison. Unfortunately, Adam's reappearance in Europe attracts the attention of his former drug dealer. Eventually, Tony kicks Adam out of his family's hotel. Adam seeks out chef-on-the-rise David (Sam Keeley), who agrees to work at Adam's restaurant and let Adam stay in his apartment. | END ID: 649

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 650 | TITLE: Venezuela | CONTENT: Shortages occur in regulated products, such as milk, various types of meat, chicken, coffee, rice, oil, precooked flour, butter prices, luxuries such as breast implants, and goods including basic necessities like toilet paper, personal hygiene products, and even medicine.[239][242][243] As a result of the shortages, Venezuelans must search for food, wait in lines for hours and sometimes settle without having certain products.[244][245] Maduro's government has blamed the shortages on "bourgeois criminals" hoarding goods.[246] | END ID: 650

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 651 | TITLE: Demosthenes | CONTENT: After Chaeronea, Philip inflicted a harsh punishment upon Thebes, but made peace with Athens on very lenient terms. Demosthenes encouraged the fortification of Athens and was chosen by the ecclesia to deliver the Funeral Oration.[101] In 337 BC, Philip created the League of Corinth, a confederation of Greek states under his leadership, and returned to Pella.[102] In 336 BC, Philip was assassinated at the wedding of his daughter, Cleopatra of Macedon, to King Alexander of Epirus. The Macedonian army swiftly proclaimed Alexander III of Macedon, then twenty years old, as the new king of Macedon. Greek cities like Athens and Thebes saw in this change of leadership an opportunity to regain their full independence. Demosthenes celebrated Philip's assassination and played a leading part in his city's uprising. According to Aeschines, "it was but the seventh day after the death of his daughter, and though the ceremonies of mourning were not yet completed, he put a garland on his head and white raiment on his body, and there he stood making thank-offerings, violating all decency."[15] Demosthenes also sent envoys to Attalus, whom he considered to be an internal opponent of Alexander.[103] Nonetheless, Alexander moved swiftly to Thebes, which submitted shortly after his appearance at its gates. When the Athenians learned that Alexander had moved quickly to Boeotia, they panicked and begged the new king of Macedon for mercy. Alexander admonished them but imposed no punishment. | END ID: 651

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 652 | TITLE: Josh Hutcherson | CONTENT: Born in Union, Kentucky, on October 12, 1992, Hutcherson is the elder son of Michelle (nÃ©e Fightmaster), a former Delta Air Lines employee who now assists with Josh's career, and Chris Hutcherson, an analyst for the United States Environmental Protection Agency (EPA).[1][2] His parents, who were also born and raised in Kentucky, met in high school in Dry Ridge.[1][3] He has one younger brother, Connor.[4][5][6] | END ID: 652

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 653 | TITLE: Rus' (name) | CONTENT: The southern territories of the Rus' fell under the rule of the Grand Duchy of Lithuania in the 13th century. While Russian descendants of the Rus' called themselves Russkiye, the residents of these lands called themselves Rusyny or Ruskiye. The name Ruthenia arises as a latinized form of the name Rus' in Western European documents at about this time. | END ID: 653

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 654 | TITLE: English versions of the Nicene Creed | CONTENT: In truth we believe in one God, God the Father the Pantocrator, maker of heaven and earth, and of all things visible and invisible. We believe in one Lord, Jesus Christ, the only begotten Son of God, begotten of the Father before all ages. Light of light, true God of true God, begotten not made, consubstantial with the Father, by whom all things came into being. This is he, who for us humans and our salvation, came down from heaven, and was incarnate of the Holy Spirit and of the Virgin Mary, and became human. And he was crucified for us under Pontius Pilate, suffered and was buried. And he rose from the dead on the third day according to the Scriptures. He ascended into the heavens and sits at the right hand of the Father. And he is also coming in his glory to judge the living and the dead, whose kingdom shall have no end. Yes, we believe in the Holy Spirit, the Lord, the giver of life, who proceeds from the Father. With the Father and the Son, we co-worship him and we co-glorify him, who spoke by the prophets. And in one holy, Catholic, and Apostolic Church. We confess one baptism for the forgiveness of sins. We look for the resurrection of the dead, and the life of the age to come. Amen.[citation needed] | END ID: 654

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 655 | TITLE: Extraocular muscles | CONTENT: The superior oblique muscle originates at the back of the orbit (a little closer to the medial rectus, though medial to it, getting rounder as it[1] courses forward to a rigid, cartilaginous pulley, called the trochlea, on the upper, nasal wall of the orbit. The muscle becomes tendinous about 10mm before it passes through the pulley, turning sharply across the orbit, and inserts on the lateral, posterior part of the globe. Thus, the superior oblique travels posteriorly for the last part of its path, going over the top of the eye. Due to its unique path, the superior oblique, when activated, pulls the eye downward and laterally.[2] | END ID: 655

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 656 | TITLE: Sapta Puri | CONTENT: of the black-headed oriole
calling pilgrims out of the dry land.
This benediction of water, overflowing. | END ID: 656

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 657 | TITLE: Fall of the Western Roman Empire | CONTENT: After the death of Olybrius there was a further interregnum until March 473, when Gundobad proclaimed Glycerius emperor. He may have made some attempt to intervene in Gaul; if so, it was unsuccessful.[209] | END ID: 657

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 658 | TITLE: Everything That Rises Must Converge | CONTENT: The title Everything That Rises Must Converge refers to a work by the French philosopher Pierre Teilhard de Chardin titled the "Omega Point":[3] "Remain true to yourself, but move ever upward toward greater consciousness and greater love! At the summit you will find yourselves united with all those who, from every direction, have made the same ascent. For everything that rises must converge."[4] | END ID: 658

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 659 | TITLE: Sonata | CONTENT: Ernest Newman wrote in the essay "Brahms and the Serpent": | END ID: 659

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 660 | TITLE: History of the world's tallest buildings | CONTENT: The earliest structures now known to be the tallest in the world were the Egyptian pyramids, with the Great Pyramid of Giza, at an original height of 146.5 metres (481 ft), being the tallest man–made structure in the world for over 3,800 years, until the construction of Lincoln Cathedral in 1300. From then until the completion of the Washington Monument (capped in 1884) the world's tallest buildings were churches or cathedrals. Later, the Eiffel Tower and, still later, some radio masts and television towers were the world's tallest structures. | END ID: 660

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 661 | TITLE: Samsung Galaxy Tab series | CONTENT: The Galaxy Tab 10.1 runs Android 3.2 Honeycomb, with Samsung's custom TouchWiz software. An update to Android 4.0 Ice Cream Sandwich is available.[9] | END ID: 661

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 662 | TITLE: Drug Enforcement Administration | CONTENT: The DEA Aviation Division or Office of Aviation Operations (OA) (formerly Aviation Section) is an airborne division based in Fort Worth Alliance Airport, Texas. The current OA fleet consists of 106 aircraft and 124 DEA pilots.[16] | END ID: 662

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 663 | TITLE: Francisca Reyes-Aquino | CONTENT: She published a thesis in 1926 entitled Philippine Folk Dances and Games where she noted on previously unrecorded forms of local celebration, ritual and sports. Her thesis was made with teachers and playground instructors from both public and private institutions in mind.[2] This work was expanded with the official support of UP President Jorge Bocobo in 1927. She then served at the university as part of the faculty for 18 years.[1] | END ID: 663

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 664 | TITLE: 2006 FIFA World Cup knockout stage | CONTENT: Man of the Match:
Gennaro Gattuso (Italy) | END ID: 664

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 665 | TITLE: Marine Hospital Service | CONTENT: The PHS Act of 1944 broadened the scope of the Commissioned Corps, allowing for the commissioning of nurses, scientists, dietitians, physical therapists, and sanitarians (later health service officers). From 1940 to 1945, the Commissioned Corps quadrupled its numbers from 625 officers to 2,600. [22] | END ID: 665

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 666 | TITLE: Lagos State | CONTENT: In 2003, many of the existing 20 LGAs were split for administrative purposes into Local Council Development Areas. These lower-tier administrative units now number 56: Agbado/Oke-Odo, Agboyi/Ketu, Agege, Ajeromi, Alimosho, Apapa, Apapa-Iganmu, Ayobo/Ipaja, Badagry West, Badagry, Bariga, Coker Aguda, Egbe Idimu, Ejigbo,
Epe, Eredo, Eti Osa East, Eti Osa West, Iba, Isolo, Imota, Ikoyi, Ibeju, Ifako-Ijaiye, Ifelodun, Igando/Ikotun, Igbogbo/Bayeku, Ijede, Ikeja, Ikorodu North, Ikorodu West, Ikosi Ejinrin, Ikorodu, Ikorodu West, Iru/Victoria Island, Itire Ikate, Kosofe, Lagos Island West, Lagos Island East, Lagos Mainland, Lekki, Mosan/Okunola, Mushin, Odi Olowo/Ojuwoye, Ojo, Ojodu, Ojokoro, Olorunda, Onigbongbo, Oriade, Orile Agege, Oshodi, Oto-Awori, Shomolu, Surulere and Yaba.[27] | END ID: 666

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 667 | TITLE: Comcast | CONTENT: In May 2008 Comcast purchased Plaxo for a reported $150 million to $170 million.[99] | END ID: 667

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 668 | TITLE: Hindu temple architecture | CONTENT: Stepped floorplan of Dattatreya Temple (one side of the shrine) with five projections at Chattarki in Gulbarga district, 12th century CE | END ID: 668

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 669 | TITLE: The Young and the Restless | CONTENT: Taped at CBS Television City, studios 41 and 43 in Hollywood since its debut on March 26, 1973,[50] the show was packaged by the distribution company Columbia Pictures Television, which has now been replaced by Sony Pictures Television.[4][51] The Young and the Restless originally aired as a half-hour series on CBS and was the first soap opera to focus on the visual aspects of production, creating "a look that broke with the visual conventions of the genre."[3][4] Similar to the radio serials that had preceded them, soap operas at the time primarily focused on dialogue, characters, and story, with details like sets as secondary concerns.[3] The Young and the Restless stood out by using unique lighting techniques and camera angles, similar to Hollywood-style productions.[51][52] The style of videotaping included using out-of-the-ordinary camera angles and a large number of facial close-ups with bright lighting on the actors' faces.[3][51][52][53] Conboy said he used lighting to create "artistic effects".[52] Those effects made the series look dark, shadowy, and moody.[3][52] The Young and the Restless' look influenced the taping styles of other soap operas.[3] When H. Wesley Kenney replaced Conboy as executive producer, he balanced the lighting of the scenes.[53] | END ID: 669

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 670 | TITLE: Hello Neighbor | CONTENT: It is implied throughout the events of the game that most of Act 3 was a nightmare occurring in the man's head, and his escape from the house signifies himself finally coming to terms with his kidnapping as a boy at the hands of the neighbor. | END ID: 670

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 671 | TITLE: Foreign trade of South Africa | CONTENT: Since the end of apartheid foreign trade in South Africa has increased, following the lifting of several sanctions and boycotts which were imposed as a means of ending apartheid. | END ID: 671

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 672 | TITLE: Proxy voting | CONTENT: Proxy voting, even if allowed, may be limited to infrequent use if the rules governing a body specify minimum attendance requirements. For instance, bylaws may prescribe that a member can be dropped for missing three consecutive meetings.[72] | END ID: 672

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 673 | TITLE: Mersenne prime | CONTENT: Besides, if we notice those prime factors, and delete "old prime factors", for example, 3 divides the 2nd, 6th, 18th, 54th, 162nd, ... terms of this sequence, we only allow the 2nd term divided by 3, if we do, they are | END ID: 673

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 674 | TITLE: Man of Constant Sorrow | CONTENT: On October 13, 2009, on the Diane Rehm Show, Ralph Stanley of the Stanley Brothers, whose autobiography is titled Man of Constant Sorrow,[9] discussed the song, its origin, and his effort to revive it:[10] | END ID: 674

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 675 | TITLE: House of Cards (UK TV series) | CONTENT: Before the series was reissued in 2013 to coincide with the release of the US version of House of Cards, Dobbs rewrote portions of the novel to bring the series in line with the television mini-series and restore continuity among the three novels.[citation needed] In the 2013 version: | END ID: 675

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 676 | TITLE: Laura Mulvey | CONTENT: Crystal Gazing exemplified more spontaneous filmmaking than their past films. Many of the elements of the film were decided once production began. The film was well received but lacked a "feminist underpinning" that had been the core of many of their past films. [7] | END ID: 676

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 677 | TITLE: Foreign relations of India | CONTENT: In the United Nations, India supported the decolonisation of Morocco and the Moroccan freedom movement. India recognised Morocco on 20 June 1956 and established relations in 1957.[426] The Ministry of External Affairs of the Government of India states that "India and Morocco have enjoyed cordial and friendly relations and over the years bilateral relations have witnessed significant depth and growth."[427] | END ID: 677

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 678 | TITLE: Metrication in Canada | CONTENT: Free trade with the United States has resulted in continued exposure to the US system. Since the United States is Canada's largest trading partner and vice versa, Canadian exporters and importers must be accustomed to dealing in US customary units as well as metric. | END ID: 678

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 679 | TITLE: Cuba | CONTENT: No political party is permitted to nominate candidates or campaign on the island, including the Communist Party.[135] The Communist Party of Cuba has held six party congress meetings since 1975. In 2011, the party stated that there were 800,000 members, and representatives generally constitute at least half of the Councils of state and the National Assembly. The remaining positions are filled by candidates nominally without party affiliation. Other political parties campaign and raise finances internationally, while activity within Cuba by opposition groups is minimal. | END ID: 679

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 680 | TITLE: Engine braking | CONTENT: The term 'engine braking' refers to the braking effect that occurs in gasoline engines when the accelerator pedal is released. | END ID: 680

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 681 | TITLE: Claudia Wells | CONTENT: Claudia Grace Wells (born July 5, 1966) is an American actress. She is best known for her role as Jennifer Parker in the film Back to the Future (1985). | END ID: 681

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 682 | TITLE: Carbon dioxide | CONTENT: Carbon dioxide is a food additive used as a propellant and acidity regulator in the food industry. It is approved for usage in the EU[34] (listed as E number E290), US[35] and Australia and New Zealand[36] (listed by its INS number 290). | END ID: 682

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 683 | TITLE: 2017 Washington train derailment | CONTENT: The NTSB interviewed the train's engineer, who suffered serious injuries, in January. He told investigators that he did not see the advance speed sign or milepost 18, mistakenly thinking he was at milepost 17. The engineer applied the train's brakes after seeing the final speed signpost, immediately north of the curve.[54] | END ID: 683

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 684 | TITLE: Sleep | CONTENT: Sleep increases the sensory threshold. In other words, sleeping persons perceive fewer stimuli. However, they can generally still respond to loud noises and other salient sensory events.[7][5] | END ID: 684

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 685 | TITLE: Ectopic pregnancy | CONTENT: Leg of fetal lamb appearing out of the uterus during caesarian section. | END ID: 685

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 686 | TITLE: Hindsight bias | CONTENT: Positive consequences of hindsight bias is an increase in one’s confidence and performance, as long as the bias distortion is reasonable and does not create overconfidence. Another positive consequence is that one’s self-assurance of their knowledge and decision-making, even if it ends up being a poor decision, can be beneficial to others; allowing others to experience new things or to learn from those who made the poor decisions.[31] | END ID: 686

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 687 | TITLE: United States Marine Corps Recruit Training | CONTENT: Close order drill is an important factor in recruit training, and begins from their first formation on the yellow footprints. In the first phase, they learn all of the basic commands and movements, memorizing the timing through the use of "ditties", or mnemonics, that help synchronize a recruit's movements with the rest of his or her platoon. Constant repetition and practice are used to facilitate muscle memory, so that any given movement can be rendered immediately and accurately upon order without hesitation. To aid in this development, drill movements are worked into other parts of daily life, to help increase the platoon's synchronization and muscle memory; this same technique is used with other non-drill activities as well. For example, a recruit is instructed to hold his/her food tray in a similar fashion to holding the butt of a rifle during "shoulder arms." | END ID: 687

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 688 | TITLE: Go West (band) | CONTENT: Peter Cox took part in TV talent competition Reborn in the USA in 2003 and came third. Following the show, Go West and winner Tony Hadley of Spandau Ballet toured together and released the album Tony Hadley Vs Peter Cox & Go West. | END ID: 688

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 689 | TITLE: List of My Little Pony Earth ponies | CONTENT: In Rainbow Rocks, Octavia plays her cello in the Battle of the Bands, but she is defeated by the Rainbooms. | END ID: 689

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 690 | TITLE: Boogie Woogie (TV series) | CONTENT: Boogie Woogie was among the first shows to start special dance championships catering to different age groups. In the first two seasons, these championships would be one to two episode long and the one winner would be decided at the end of every episode. | END ID: 690

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 691 | TITLE: Psychoanalysis | CONTENT: Psychoanalysis is a set of theories and therapeutic techniques[1] related to the study of the unconscious mind,[2] which together form a method of treatment for mental-health disorders. The discipline was established in the early 1890s by Austrian neurologist Sigmund Freud and stemmed partly from the clinical work of Josef Breuer and others. | END ID: 691

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 692 | TITLE: Process area (CMMI) | CONTENT: Specific Practices by Goal | END ID: 692

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 693 | TITLE: Coolant | CONTENT: In some applications, solid materials are used as coolants. The materials require high energy to vaporize; this energy is then carried away by the vaporized gases. This approach is common in spaceflight, for ablative atmospheric reentry shields and for cooling of rocket engine nozzles. The same approach is also used for fire protection of structures, where ablative coating is applied. | END ID: 693

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 694 | TITLE: Zither | CONTENT: Anton Karas and Ruth Welcome used instruments of similar design to the one illustrated. After World War II, Karas (according to zither scholar Günter Wittenstein, who was acquainted with him) performed on an instrument of larger dimensions than normal – with a 43 cm standard scale length for the fingerboard strings. He used Viennese tuning (see below), but with an altered chromatic sequence for the fingerboard and open strings. The accompaniment strings G and F♯ were tuned an octave higher, while contrabass strings tuned E♭, F, D, E, C♯ replaced the regular cycle of fifths bass strings. This brought the contrabasses closer to the fingerboard where the player could reach them more easily. | END ID: 694

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 695 | TITLE: P versus NP problem | CONTENT: Although it is unknown whether P = NP, problems outside of P are known. A number of succinct problems (problems that operate not on normal input, but on a computational description of the input) are known to be EXPTIME-complete. Because it can be shown that P ≠ EXPTIME, these problems are outside P, and so require more than polynomial time. In fact, by the time hierarchy theorem, they cannot be solved in significantly less than exponential time. Examples include finding a perfect strategy for chess (on an N × N board)[15] and some other board games.[16] | END ID: 695

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 696 | TITLE: Jaylen Brown | CONTENT: Brown's father is Quenton M. Brown, a professional boxer, who is the 2016 WBU World Champion, the 2015 WBU C.A.M. Heavyweight Champion, and a member of the Hawaii State Boxing Commission Board.[39] | END ID: 696

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 697 | TITLE: Galvanic cell | CONTENT: Volta was the inventor of the voltaic pile, the first electrical battery. In common usage, the word "battery" has come to include a single galvanic cell, but a battery properly consists of multiple cells.[1] | END ID: 697

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 698 | TITLE: Pink Floyd live performances | CONTENT: Thanks to stage architect/designer Mark Fisher, Pink Floyd's tours became a staple in the industry because of their outstanding special and scenic effects. Pyrotechnics (such as exploding flashpots, an exploding gong and fireworks) and dry ice were used extensively throughout Pink Floyd's career. In 1973's tour to promote The Dark Side of the Moon, a large scale model plane flew over the audience and crashed onto the stage with a spectacular explosion, an effect repeated at the start of The Wall and the Division Bell shows. During shows to promote A Momentary Lapse of Reason, a similar effect was achieved with a flying bed. | END ID: 698

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 699 | TITLE: NATO | CONTENT: The meetings of the North Atlantic Council are chaired by the Secretary General of NATO and, when decisions have to be made, action is agreed upon on the basis of unanimity and common accord. There is no voting or decision by majority. Each nation represented at the Council table or on any of its subordinate committees retains complete sovereignty and responsibility for its own decisions. | END ID: 699

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 700 | TITLE: White Army, Black Baron | CONTENT: The tune was also used for communist songs in other languages, including Weimar Germany in the 1920s by German Communists. An early German version with the incipit German: WeiÃŸes Gesindel und adlige Brut ("White riffraff, noble scum") was a free translation of the original lyrics:[1] | END ID: 700

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 701 | TITLE: Thales of Miletus | CONTENT: The view that all matter is one is quite a reputable scientific hypothesis. | END ID: 701

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 702 | TITLE: 2018 FIFA World Cup qualification – UEFA Group E | CONTENT: Romania  v  Kazakhstan | END ID: 702

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 703 | TITLE: Paramount Pictures | CONTENT: On December 11, 2005, the Paramount Motion Pictures Group announced that it had purchased DreamWorks SKG (which was co-founded by former Paramount executive Jeffrey Katzenberg) in a deal worth $1.6Â billion. The announcement was made by Brad Grey, chairman and CEO of Paramount Pictures who noted that enhancing Paramount's pipeline of pictures is a "key strategic objective in restoring Paramount's stature as a leader in filmed entertainment."[63] The agreement does not include DreamWorks Animation SKG Inc., the most profitable part of the company that went public the previous year.[64] | END ID: 703

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 704 | TITLE: Filename extension | CONTENT: When the Internet age first arrived, those using Windows systems that were still restricted to 8.3 filename formats had to create web pages with names ending in .HTM, while those using Macintosh or UNIX computers could use the recommended .html filename extension. This also became a problem for programmers experimenting with the Java programming language, since it requires source code files to have the four-letter suffix .java and compiles object code output files with the five-letter .class suffix.[5] | END ID: 704

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 705 | TITLE: Symphony No. 101 (Haydn) | CONTENT: The Symphony No. 101 in D major (Hoboken 1/101) is the ninth of the twelve London symphonies written by Joseph Haydn. It is popularly known as The Clock because of the "ticking" rhythm throughout the second movement. | END ID: 705

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 706 | TITLE: Kauvery Hospital | CONTENT: Dr Manivannan S (Joint Managing Director) | END ID: 706

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 707 | TITLE: Screen Actors Guild Award for Outstanding Performance by a Female Actor in a Leading Role | CONTENT: (21st) | END ID: 707

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 708 | TITLE: The Blitz | CONTENT: In the 1920s and 1930s, air power theorists like Giulio Douhet and Billy Mitchell claimed that air forces could win wars, obviating the need for land and sea fighting.[12] It was thought that bombers would always get through and could not be resisted, particularly at night. Industry, seats of government, factories and communications could be destroyed, depriving an opponent of the means to make war. Bombing civilians would cause a collapse of morale and a loss of production in the remaining factories. Democracies, where public opinion was allowed, were thought particularly vulnerable. The RAF and the United States Army Air Corps (USAAC) adopted much of this apocalyptic thinking. The policy of RAF Bomber Command became an attempt to achieve victory through the destruction of civilian will, communications and industry.[13] | END ID: 708

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 709 | TITLE: Walls of Constantinople | CONTENT: The next gate, Yeni Ayakapı ("New Gate of the Saint"), is not Byzantine, unless it replaces an earlier Byzantine entrance.[165] It was constructed by the great Ottoman architect Mimar Sinan in 1582.[166] Shortly after it lies the older Ayakapı ("Gate of the Saint"), known in Greek as the St. Theodosia Gate (Πύλη τῆς Ἁγίας Θεοδοσίας) after the great earby church of St. Theodosia (formerly identified with the Gül Mosque).[165] The next gate is that of Eis Pegas (Πύλη εἰς Πηγάς, Pylē eis Pēgas), known by Latin chroniclers as Porta Puteae or Porta del Pozzo, modern Cibali Kapısı. It was named so because it looked towards the quarter of Pegae (Πηγαὶ, Pēgai, "springs") on the other shore of the Golden Horn.[167] Next was the now-demolished Gate of the Platea (Πόρτα τῆς Πλατέας, Porta tēs Plateas) follows, rendered as Porta della Piazza by Italian chroniclers, and called in Turkish Unkapanı Kapısı ("Gate of the Flour Depot"). It was named after the local quarter of Plate[i]a ("broad place", signifying the broad shoreline at this place).[168] The next gate, Ayazma Kapısı ("Gate of the Holy Well"), is in all probability an Ottoman-era structure.[169] | END ID: 709

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 710 | TITLE: List of Rugrats episodes | CONTENT: The series premiered on Sunday, August 11, 1991, as the second Nicktoon after Doug and preceding The Ren & Stimpy Show. Production initially halted in 1993 after 65 episodes, with the last episode airing on May 22, 1994. From 1995 to 1996, the only new episodes broadcast were "A Rugrats Passover" and "A Rugrats Chanukah", two Jewish-themed episodes that received critical acclaim; during this time, well-after the end of the show's production run, Rugrats began to receive a boost in ratings and popularity, due to constant reruns on Nickelodeon. In 1996, Klasky Csupo Animation began producing new episodes, and the show's fourth season began airing in 1997. As a result of the show's popularity, a series of theatrical films were released; The Rugrats Movie, which introduced Tommy's younger brother Dil, was released in 1998, Rugrats in Paris: The Movie, which introduced Kimi and Kira, released in 2000, and Rugrats Go Wild, a crossover film with fellow Klasky Csupo series The Wild Thornberrys, released in 2003. The final episode aired on August 1, 2004,[3] bringing the series to a total of 172 episodes and 9 seasons during a 12-year run. | END ID: 710

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 711 | TITLE: Boston Tea Party | CONTENT: The Boston Tea Party arose from two issues confronting the British Empire in 1765: the financial problems of the British East India Company; and an ongoing dispute about the extent of Parliament's authority, if any, over the British American colonies without seating any elected representation. The North Ministry's attempt to resolve these issues produced a showdown that would eventually result in revolution.[5] | END ID: 711

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 712 | TITLE: Aesop Rock | CONTENT: In February 2005, Aesop Rock released a new EP, Fast Cars, Danger, Fire and Knives. The first pressing of the EP included an 88-page booklet with lyrics from every release from Float until this EP (the lyric booklet is titled The Living Human Curiosity Sideshow); later pressings of the album come without the booklet, but with an additional bonus track, "Facemelter". In addition, a limited number of albums were available direct from Def Jux with Aesop Rock's graffiti tag on them. In response to demands from his fans, Rock did less production on the EP: three songs are produced by Blockhead, three produced by Aesop, and one by Rob Sonic. During this time he was asked to join The Weathermen to replace Vast Aire. | END ID: 712

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 713 | TITLE: 2009 FA Cup Final | CONTENT: Everton were without long-term injury victims Phil Jagielka, Mikel Arteta, Yakubu Aiyegbeni, Victor Anichebe and Nuno Valente. Andy van der Meyde, who set up the winning goal in the fourth round tie with Liverpool, had since been released by the club. This meant that there was a place on the Everton bench for 17-year-old winger Jose Baxter. | END ID: 713

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 714 | TITLE: J'son (comics) | CONTENT: Emperor Jason of Spartax first appeared in Marvel Preview #11 and was created by Chris Claremont and John Byrne. Emperor Jason was the father of the Star-Lord character who had been introduced in Marvel Preview #4. | END ID: 714

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 715 | TITLE: Fatwa | CONTENT: In 2012, the Indonesian Ulema Council issued an edict for Muslims not to wish Christians a happy Christmas. The edict said that wishing a happy Christmas was akin to confirming the "misguided" teachings of Christianity.[55] | END ID: 715

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 716 | TITLE: Cadillac Mountain | CONTENT: Driving or hiking to the summit of Cadillac Mountain to see "the nation's first sunrise" is a popular activity among visitors of Acadia National Park. However, Cadillac only sees the first sunrise in the fall and winter, when the sun rises south of due east.[5] During most of the spring and summer, the sun rises first on Mars Hill, 150 miles (240Â km) to the northeast. For a few weeks around the equinoxes, the sun rises first at West Quoddy Head in Lubec, Maine. | END ID: 716

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 717 | TITLE: Eurovision Song Contest 2016 | CONTENT: Below is a summary of the maximum 12 points award by each country's professional jury in the second semi-final: | END ID: 717

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 718 | TITLE: IP address | CONTENT: The IP address space is managed globally by the Internet Assigned Numbers Authority (IANA), and by five regional Internet registries (RIR) responsible in their designated territories for assignment to end users and local Internet registries, such as Internet service providers. IPv4 addresses have been distributed by IANA to the RIRs in blocks of approximately 16.8Â million addresses each. Each ISP or private network administrator assigns an IP address to each device connected to its network. Such assignments may be on a static (fixed or permanent) or dynamic basis, depending on its software and practices. | END ID: 718

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 719 | TITLE: New York Knicks | CONTENT: The rivalry between the New York Knicks and the Indiana Pacers started in 1993 and quickly became one of the most bitter in NBA history. They met in the playoffs 6 times from 1993 to 2000, fueling a rivalry epitomized by the enmity between Reggie Miller and prominent Knick fan Spike Lee. Miller likened it to the Hatfield–McCoy feud, and The New York Times said in 1998 that it was "as combustible as any in the league". The rivalry gave Miller the nickname "The Knick-Killer". His clutch performances were frequently followed by jabs at Lee like the choke sign, adding fuel to the rivalry. The rivalry renewed during the 2013 NBA Playoffs in the Eastern Conference Semifinals, with Indiana taking the series 4 games to 2. | END ID: 719

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 720 | TITLE: Trial by combat | CONTENT: In c. 1219 trial by jury replaced trial by ordeal, which had been the mode of proof for crown pleas since the Assize of Clarendon in 1166. With the emergence of the legal profession in the thirteenth century, lawyers, guarding the safety of the lives and limbs of their clients, steered people away from the wager of battle. A number of legal fictions were devised to enable litigants to avail themselves of the jury even in the sort of actions that were traditionally tried by wager of battle. The practice of averting trial by combat led to the modern concept of attorneys representing litigants. | END ID: 720

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 721 | TITLE: Panama Canal expansion project | CONTENT: The growth in usage of the Panama Canal over the past few years has been almost entirely driven by increased US imports from China passing through the canal en route to ports on the US East and Gulf coasts. But it is increasingly recognized in both the US and China that this imbalance in trade is unsustainable and will be reduced via some sort of adjustment in the coming years[17] (although such an imbalance need not be made up by physically shipped goods, but could be made by other trade such as intellectual property as China upgrades its intellectual property protection laws). The ACP, however, presumes that trade will continue to grow for a generation as it has for the past several years.[citation needed] | END ID: 721

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 722 | TITLE: List of Major League Baseball career WHIP leaders | CONTENT: Below is the list of the top 100 Major League Baseball pitchers in Walks plus hits per inning pitched (WHIP) with at least 1,000 innings pitched. | END ID: 722

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 723 | TITLE: Varna (Hinduism) | CONTENT: Peter Masefield,[34] a Buddhism scholar and ancient Pali texts translator, states that during the Nikāya texts period of Buddhism (3rd century BC to 5th century AD), Varna as a class system is attested, but the described Varna was not a caste system. The Pali texts enumerate the four Varnas Brahmin, "Kshatriya",Vessa (Vaishya) and Sudda (Shudra).[34] Masefield notes that people in any Varna could in principle perform any profession. The early Buddhist texts, for instance, identify some Brahmins to be farmers and in other professions. The text state that anyone, of any birth, could perform the priestly function,[34] and that the Brahmin took food from anyone, suggesting that strictures of commensality were as yet unknown. The Nikaya texts also imply that endogamy was not mandated in ancient India. Masefield concludes, "if any form of caste system was known during the Nikaya period - and it is doubtful that it was - this was in all probability restricted to certain non-Aryan groups".[34] | END ID: 723

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 724 | TITLE: Jason Gideon | CONTENT: In the season ten episode "Nelson's Sparrow", Gideon was murdered off-screen, having been shot dead at a close range by a serial killer named Donnie Mallick (Arye Gross), which prompts the BAU team to investigate Gideon's murder. During the flashbacks focusing on a young version of him for the episode which show him working at the BAU in 1978, he is played by Ben Savage. | END ID: 724

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 725 | TITLE: Degrassi (season 14) | CONTENT: ^Note 2Â : Although Ehren Kassam was not credited with the main cast in the credits, Bell Media and Epitome still considered him a series regular.[8] | END ID: 725

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 726 | TITLE: Maleficent (film) | CONTENT: Linda Woolverton's screenplay went through at least 15 versions as the film progressed in the production.[20] Director Robert Stromberg said: "I met many times with Linda Woolverton, the writer. We did lots of roundtable discussions and sort of cut out the fat as much as we could and sort of purified the storyline as much as we could".[21] In some earlier versions of the story, Stefan was the half-human, half-fairy bastard son of King Henry. The version of the screenplay which went into shooting originally included two characters called Queen Ulla and King Kinloch, the fairy queen and the fairy king of the Moors, and the aunt and uncle of Maleficent.[5] Miranda Richardson and Peter Capaldi were cast and shot the Queen Ulla and King Kinloch scenes, but their roles were cut in the editing process together with more than 15 minutes of the first act of the film. Stromberg said: "We spent a bit more time originally in the fairy world before we got into the human side of things ... we wanted to get it [the film] under two hours. So we cut about fifteen minutes out of the first act, and then that had to be seamed together with some pretty basic reshoots."[22] | END ID: 726

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 727 | TITLE: Shrek The Musical | CONTENT: The original principal casts of the English-speaking productions. | END ID: 727

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 728 | TITLE: Marine pollution | CONTENT: Estuaries tend to be naturally eutrophic because land-derived nutrients are concentrated where runoff enters the marine environment in a confined channel. The World Resources Institute has identified 375 hypoxic coastal zones around the world, concentrated in coastal areas in Western Europe, the Eastern and Southern coasts of the US, and East Asia, particularly in Japan.[43] In the ocean, there are frequent red tide algae blooms[44] that kill fish and marine mammals and cause respiratory problems in humans and some domestic animals when the blooms reach close to shore. | END ID: 728

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 729 | TITLE: Vicks VapoRub | CONTENT: Richardson-Vicks was sold to Procter & Gamble in 1985 and is now known as Vicks. VapoRub is also currently manufactured and packaged in India and Mexico. In German-speaking countries (the exception of Switzerland) it is sold under the name Wick VapoRub.[1] VapoRub continues to be Vicks's flagship product internationally, and the Vicks brand name is often used synonymously with the VapoRub product. | END ID: 729

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 730 | TITLE: Community college | CONTENT: TAFEs and other providers carry on the tradition of adult education, which was established in Australia around the mid 19th century when evening classes were held to help adults enhance their numeracy and literacy skills.[1] Most Australian universities can also be traced back to such forerunners, although obtaining a university charter has always changed their nature. In TAFEs and colleges today, courses are designed for personal development of an individual and/or for employment outcomes. Educational programs cover a variety of topics such as arts, languages, business and lifestyle, and are usually timetabled to run two, three or four days of the week, depending on the level of the course undertaken. A Certificate I may only run for 4 hours twice a week for a term of 9 weeks. A full-time Diploma course might have classes 4 days per week for a year (36 weeks). Some courses may be offered in the evenings or weekends to accommodate people working full-time. Funding for colleges may come from government grants and course fees, and many are not-for-profit organisations. There are located in metropolitan, regional and rural locations of Australia. | END ID: 730

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 731 | TITLE: Emma Swan | CONTENT: After Emma and Hook are dragged into Zelena's time portal, she accepts Storybrooke as her home, regaining her magic to re-open the portal to the present where she begins a relationship with Hook. However, unaware to her, she also brings a previously deceased Maid Marian to the future which ruins Regina's happiness, having not heeded the warning of messing with the past. Elsa, who was trapped in an urn, was also brought to Storybrooke by the time portal. While working with Elsa, who helps Emma to finally embrace and control her powers, Emma helps her new friend find her sister and return home while balancing the threat of the Snow Queen against her friends and family. After a period of peace, Emma begins to help Regina on her quest to find the Author of Henry's book. Cruella De Vil and Ursula soon come into town resurrecting Maleficent and working with Rumplestiltskin to find the Author of the magical tome Once Upon a Time. After Cruella De Vil threatens to kill Henry, Emma kills her, soon after learning of her parents' actions of removing Emma's potential for darkness by putting black magic within Maleficent's daughter and Emma's childhood friend Lily.[39] After Emma returns to town, with the encouragement from Hook, she chooses to forgive her parents and let go of her anger. In moments before the finale, Emma is able to finally let her walls down and tells Hook she loves him. Emma then chooses to sacrifice herself for the town of Storybrooke, asking her parents and Hook to save her, she voluntarily plunges the dagger into the Darkness, transforming into the new Dark One. | END ID: 731

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 732 | TITLE: Philadelphia 76ers | CONTENT: In the 1997–98 season, the Sixers drastically changed their logo and colors in an effort to appeal to a more youthful, hip-hop oriented culture. The iconic 76 logo was dropped, and a new logo was introduced, featuring a bigger 76ers script, with a single star behind the number 7 and a streaking basketball below. More controversially, gold and black were introduced to the color scheme, along with red, white and blue. Uniforms were primarily white (home), and black (away), with slight adjustments in the home logo lettering (gold 1997–2000, black 2000–09), trim and piping. Until the 2006–07 season, player names featured a red trim, before dropping it altogether and shrunk the font size in the 2007–08 season; the alternates adopted this design that season, with the regular uniforms following suit that same season. A blue alternate uniform was worn 1999–2006, while a red alternate uniform, featuring a return to the 'PHILA' script in then-current lettering, was worn 2006–09. This logo and color scheme were used until the 2008–09 season. | END ID: 732

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 733 | TITLE: Semiconservative replication | CONTENT: The deciphering of the structure of DNA by Watson and Crick in 1953 suggested that each strand of the double helix would serve as a template for synthesis of a new strand. However, there was no way of knowing how the newly synthesized strands might combine with the template strands to form two double helical DNA molecules. The semiconservative model seemed most reasonable since it would allow each daughter strand to remain associated with its template strand. The semiconservative model was supported by the Meselson-Stahl experiment[2][3] and other even more revealing experiments that allowed for autoradiographic visualization of the distribution of old and new strands within replicated chromosomes. | END ID: 733

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 734 | TITLE: Flag of Hungary | CONTENT: By a government decree from 2000, the ratio (which is neither defined in the Constitution nor in 1995[10] or 2000[11] legislation) of flags used on government building is 1:2. | END ID: 734

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 735 | TITLE: Mediterranean forests, woodlands, and scrub | CONTENT: A mediterranean cork oak, in Alentejo region, Portugal. | END ID: 735

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 736 | TITLE: List of largest poker tournaments in history (by prize pool) | CONTENT: Below are the 30 largest poker tournaments with respect to the prize pool in United States dollars and not number of entrants. This list includes live and online poker. | END ID: 736

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 737 | TITLE: Religious debates over the Harry Potter series | CONTENT: Regardless, statements such as those in Witchcraft Repackaged that the books depict actual occultist practices of any kind have been roundly criticised. Christian writer Stephen D. Greydanus writes that the magic of the Harry Potter novels is not the ritualistic, invocative magic of Wicca or occultism but the same "fantasy" magic practised in the works of J. R. R. Tolkien and C. S. Lewis; "If anything, the magic in Rowling's world is even more emphatically imaginary, even further removed from real-world practices, than that of Tolkien or Lewis; and, like theirs, presents no appreciable risk of direct imitative behaviour."[107] Christianity Today columnist Charles Colson asserts that the magic in Harry Potter is "purely mechanical, as opposed to occultic. That is, Harry and his friends cast spells, read crystal balls, and turn themselves into animalsâ€”but they don't make contact with a supernatural world. [It's not] the kind of real-life witchcraft the Bible condemns."[3] Austin Cline notes that, "The Harry Potter books simply aren't about Wicca as it is currently practiced. J.K Rowling researched Wiccan practices and incorporated a few elements in order to give her books a bit more of an air of reality, but she and Wicca are drawing upon the same corpus of ancient traditions and stories so similarities are inevitable. They certainly aren't a sign that the books work to "indoctrinate" people into Wicca as a religion."[108] | END ID: 737

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 738 | TITLE: The West Wing | CONTENT: While several real-world leaders exist in the show's universe, most foreign countries depicted or referred to on the show have fictional rulers. Real people mentioned in The West Wing include Muammar Gaddafi, Yasser Arafat, Fidel Castro, Queen Elizabeth II, King Bhumibol Adulyadej, King Carl XVI Gustaf, Thabo Mbeki and Osama bin Laden. However, when a peace accord is worked out between Israel and the Palestinian Authority at the start of the show's sixth season, the Chairman of the Palestinian Authority is the fictional Nizar Farad, not Yasser Arafat. (By that time in the real world, Arafat was dead and a successor, Rawhi Fattuh, had been elected.) | END ID: 738

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 739 | TITLE: Matthew 6:26 | CONTENT: For a collection of other versions see BibRef Matthew 6:26 | END ID: 739

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 740 | TITLE: Gunga Din | CONTENT: The poem is a rhyming narrative from the point of view of an English soldier in India, about an Indian water-bearer (a bhishti) who saves the soldier's life but is soon shot and killed. In the final three lines, the soldier regrets the abuse he dealt to Din and admits that Din is the better man of the two. The poem was published as one of the set of martial poems called the Barrack-Room Ballads. | END ID: 740

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 741 | TITLE: Operations management for services | CONTENT: For manufactured products, quality problems are handled through warranties, returns and repair after the product is delivered. In high contact services there is no time to fix quality problems later; they must be handled by service recovery as the service is delivered. For example, if soup is spilled on the customer in a restaurant, the waiter might apologize, offer to pay to have the suit cleaned and provide a free meal. If a hotel room is not ready when promised, the staff could apologize, offer to store the customer's luggage or provide an upgraded room. Service recovery is intended to fix the problem on the spot and go even further to offer the customer some form of consolation and compensation. The objective is to make the customer satisfied with the situation, even though there was a service failure.[30][31] | END ID: 741

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 742 | TITLE: Lynn Cartwright | CONTENT: Lynn Cartwright (February 27, 1927 â€“ January 2, 2004) was an American character actress known for her performance as the older version of Geena Davis' character, Dottie Hinson, in the 1992 film A League of Their Own. | END ID: 742

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 743 | TITLE: Forty-second Amendment of the Constitution of India | CONTENT: Almost all parts of the Constitution, including the Preamble and amending clause, were changed by the 42nd Amendment, and some new articles and sections were inserted. The amendment's fifty-nine clauses stripped the Supreme Court of many of its powers and moved the political system toward parliamentary sovereignty. It curtailed democratic rights in the country, and gave sweeping powers to the Prime Minister's Office.[3] The amendment gave Parliament unrestrained power to amend any parts of the Constitution, without judicial review. It transferred more power from the state governments to the central government, eroding India's federal structure. The 42nd Amendment also amended the Preamble and changed the description of India from "sovereign democratic republic" to a "sovereign, socialist secular democratic republic", and also changed the words "unity of the nation" to "unity and integrity of the nation". | END ID: 743

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 744 | TITLE: Reykjavík Summit | CONTENT: The talks finally stalled, Reagan asking if Gorbachev would "turn down a historic opportunity because of a single word," referring to his insistence on laboratory testing.  Gorbachev asserted that it was a matter of principle, and the summit concluded. | END ID: 744

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 745 | TITLE: Consultant | CONTENT: Similarly, the growth of online, highly skilled consultant marketplaces has begun to grow.[4] These online platforms provide consultants with experience working for typical consulting firms to easily transition into freelancing. This means that many consultants have become much more flexible in where they can work and the nature of their work. | END ID: 745

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 746 | TITLE: Hawaii | CONTENT: The history of Hawaii's economy can be traced through a succession of dominant industries; sandalwood,[140] whaling,[141] sugarcane, pineapple, the military, tourism and education. Since statehood in 1959, tourism has been the largest industry, contributing 24.3% of the gross state product (GSP) in 1997, despite efforts to diversify. The state's gross output for 2003 was US$47Â billion; per capita income for Hawaii residents in 2014 was US$54,516.[142] Hawaiian exports include food and clothing. These industries play a small role in the Hawaiian economy, due to the shipping distance to viable markets, such as the West Coast of the contiguous U.S. The state's food exports include coffee, macadamia nuts, pineapple, livestock, sugarcane and honey.[143] | END ID: 746

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 747 | TITLE: Michael C. Hall | CONTENT: He assumed the title role in Hedwig and The Angry Inch on Broadway on October 16, 2014 and performed the role until January 18, 2015. Hall returned to the role of Hedwig from February 17â€“21, 2015 to replace John Cameron Mitchell, who had a knee injury. | END ID: 747

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 748 | TITLE: My Pillow | CONTENT: My Pillow, Inc., is a pillow manufacturing company based in Chaska, Minnesota, United States.[1] The company was founded in 2004 by Michael J Lindell, who invented and patented MyPillow, an open-cell, poly-foam pillow design. My Pillow has sold over 41 million pillows, due mostly to My Pillowâ€™s TV infomercials.[2]
[3] The company started with five employees in 2004 and had 1,500 employees as of 2017.[4] | END ID: 748

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 749 | TITLE: United States presidential election, 1884 | CONTENT: The Anti-Monopoly National Convention assembled in the Hershey Music Hall in Chicago, Illinois. The party had been formed to express opposition to the business practices of the emerging nationwide companies. There were around 200 delegates present from 16 states, but 61 of those delegates had come from Michigan and Illinois. | END ID: 749

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 750 | TITLE: French Guiana | CONTENT: French Guiana lies between latitudes 2° and 6° N, and longitudes 51° and 55° W. It consists of two main geographical regions: a coastal strip where the majority of the people live, and dense, near-inaccessible rainforest which gradually rises to the modest peaks of the Tumuc-Humac mountains along the Brazilian frontier. French Guiana's highest peak is Bellevue de l'Inini in Maripasoula (851 m (2,792 ft)). Other mountains include Mont Machalou (782 m (2,566 ft)), Pic Coudreau (711 m (2,333 ft)) and Mont St Marcel (635 m (2,083 ft)), Mont Favard (200 m (660 ft)) and Montagne du Mahury (156 m (512 ft)). | END ID: 750

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 751 | TITLE: Khmer Rouge rule of Cambodia | CONTENT: Article 20 of the 1976 Constitution of Democratic Kampuchea guaranteed religious freedom, but it also declared that "all reactionary religions that are detrimental to Democratic Kampuchea and the Kampuchean People are strictly forbidden." About 85 percent of the population follows the Theravada school of Buddhism. The country's 40,000 to 60,000 Buddhist monks, regarded by the regime as social parasites, were defrocked and forced into labour brigades. | END ID: 751

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 752 | TITLE: History of South Africa | CONTENT: Indian slaves from the Dutch colonies had been introduced into the Cape area of South Africa by the Dutch settlers in 1654.[52] By 1860, with slavery having been abolished in 1834, and after the annexation of Natal as a British colony in 1843, the British colonialists in Natal (now kwaZulu-Natal) turned to India to resolve a labour shortage. Men of the local Zulu warrior nation were refusing to adopt the servile position of labourers. In that year, the SS Truro arrived in Durban harbour with over 300 Indians on board. Over the next 50 years, 150,000 more indentured Indian servants and labourers arrived, as well as numerous free "passenger Indians", building the base for what would become the largest Indian community outside India. By 1893, when the lawyer and social activist Mahatma Gandhi arrived in Durban, Indians outnumbered whites in Natal. The civil rights struggle of Gandhi's Natal Indian Congress failed; until the 1994 advent of democracy, Indians in South Africa were subject to most of the discriminatory laws that applied to all non-white inhabitants of the country. | END ID: 752

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 753 | TITLE: List of General Hospital characters | CONTENT: Joe Scully, Jr. (deceased) | END ID: 753

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 754 | TITLE: Project Runway (season 9) | CONTENT: Original Airdate: October 13, 2011 | END ID: 754

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 755 | TITLE: Krypton-85 | CONTENT: It has a half-life of 10.756 years and a maximum decay energy of 687 keV.[1] It decays into stable, non-radioactive rubidium-85. Its most common decay (99.57%) is by beta particle emission with maximum energy of 687 keV and an average energy of 251 keV. The second most common decay (0.43%) is by beta particle emission (maximum energy of 173 keV) followed by gamma ray emission (energy of 514 keV).[2] Other decay modes have very small probabilities and emit less energetic gammas.[1][3] There are 33 other known isotopes of krypton. | END ID: 755

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 756 | TITLE: Andromeda–Milky Way collision | CONTENT: Excluding planetary engineering, by the time the two galaxies collide the surface of the Earth will have already become far too hot for liquid water to exist, ending all terrestrial life; that is currently estimated to occur in about 3.75 billion years due to gradually increasing luminosity of the Sun (it will have risen by 35â€“40% above the current luminosity).[14][15] | END ID: 756

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 757 | TITLE: 2011 Christchurch earthquake | CONTENT: The Forsyth Barr Building survived the earthquake but many occupants were trapped after the collapse of the stairwells, forcing some to abseil out after the quake.[39] Search of the building was technically difficult for USAR teams, requiring the deconstruction of 4-tonne stair sets, but the building was cleared with no victims discovered.[40] | END ID: 757

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 758 | TITLE: Geography of Somalia | CONTENT: The Jubba River enters the Indian Ocean at Kismaayo. Although the Shabeelle River at one time apparently also reached the sea near Merca, its course is thought to have changed in prehistoric times. The Shabeelle now turns southwestward near Balcad (about thirty kilometers north of Mogadishu) and parallels the coast for more than eighty-five kilometers. The river is perennial only to a point southwest of Mogadishu; thereafter it consists of swampy areas and dry reaches and is finally lost in the sand east of Jilib, not far from the Jubba River. During the flood seasons, the Shabeelle River may fill its bed to a point near Jilib and occasionally may even break through to the Jubba River farther south. Favorable rainfall and soil conditions make the entire riverine region a fertile agricultural area and the center of the country's largest sedentary population. | END ID: 758

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 759 | TITLE: Soul Food (film) | CONTENT: Soul Food is told through the eyes of 11-year-old Ahmad (Hammond), following the trials of the Joseph family, a close-knit Chicago family that gets together to have Sunday dinner every week, with plenty of soul food to go around. Mother (Big Mama) Joe (Hall) has three daughters, who each have had varying success in life: oldest daughter Teri (Williams) has become a successful lawyer, but has a strained relationship with younger sister Maxine (Fox) who stole and eventually married Teri's former boyfriend, Kenny (Sams). Teri is currently married to Miles (Beach), a lawyer who quit his job to pursue his dream of being an R&B musician, which Teri doesn't support. Youngest daughter Robin (Long)—nicknamed "Bird"—has just opened a barbershop/beauty parlor and gotten married to Lem (Phifer), an ex-convict. | END ID: 759

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 760 | TITLE: Vikram Samvat | CONTENT: Vikram Samvat (Hindi: विक्रम सम्वत्, Nepali: विक्रम सम्वत्) (abbreviated as V.S. (or VS) or B.S. (or BS));  Listen (help·info))is the historical Hindu calendar of India and Nepal. It uses lunar months and solar sidereal year (see: Vedic time keeping). It is used as the official calendar in Nepal. | END ID: 760

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 761 | TITLE: Medina | CONTENT: Throughout the winter and spring of 623 other raiding parties were sent by Muhammad from Medina. | END ID: 761

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 762 | TITLE: Mahatma Gandhi | CONTENT: In his early years, the former President of South Africa Nelson Mandela was a follower of the nonviolent resistance philosophy of Gandhi.[370] Bhana and Vahed commented on these events as "Gandhi inspired succeeding generations of South African activists seeking to end White rule. This legacy connects him to Nelson Mandela...in a sense Mandela completed what Gandhi started."[373] | END ID: 762

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 763 | TITLE: Number Eight (Battlestar Galactica) | CONTENT: In the episode "Kobol's Last Gleaming", Commander Adama sends Boomer on a mission to destroy the basestar orbiting Kobol. On the basestar, she encounters numerous other Number Eight copies identical to herself but she refuses to accept she is a Cylon and personally sets the bomb. After returning to Galactica, her hidden programming takes over and she shoots Commander Adama twice in the chest, putting him in a coma. She is put in the brig and violently interrogated by Colonel Tigh, who has taken command of Galactica during Adama's incapacity. Tyrol is also suspected of being a Cylon because of his relationship with her, and is thrown in her cell. He tells her not to speak to him or even touch him; he insists she is a machine and nothing like him. | END ID: 763

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 764 | TITLE: Microchip implant (animal) | CONTENT: A microchip implant is a passive RFID device. Lacking an internal power source, it remains inert until it is powered by the scanner. | END ID: 764

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 765 | TITLE: Ontario Highway 407 | CONTENT: Immediately east of Brock Road, this tollway falls under the ownership of the Province of Ontario and is now referred to as Ontario Highway 407 (Or Highway 407 East) instead of 407 ETR. This route runs parallel to both Highway 7 and Durham Regional Road 3 (with some crossovers) through the North of Pickering, Whitby, and Oshawa, until its eastern terminus at the proposed interchange between Highway 418 and Taunton Road[20]. A major interchange of this route includes with Highway 412, which is a spur connecting the 407 with Highway 401 in Whitby. Both the 407 East Extension, as far as Harmony Road in Oshawa. and Highway 412 opened to traffic on June 20, 2016.[21] The highway will be further extended eastward through Clarington by 2020. The tolls along this portion of the highway began on February 1, 2017. [22] | END ID: 765

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 766 | TITLE: Bosom of Abraham | CONTENT: The concept of paradise is not mentioned in Luke 16, nor are any of the distinguishing Jewish associations of paradise such as Third Heaven (found with "paradise" in 2 Corinthians 12:2–4 and Apocalypse of Moses), or the tree of life (found with "paradise" in Genesis 2:8 Septuagint and Book of Revelation 2:7).[18] Consequently, identification of Bosom of Abraham with Paradise is contested.[19] It is not clear whether Matthew 8:11 "And I tell you that many will come from the East and West and will eat with Abraham, Isaac, and Jacob in the kingdom of heaven." represents an alternative or complimentary cosmology to the ideas of Luke 16:19–31.[20] | END ID: 766

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 767 | TITLE: History of Chinese Australians | CONTENT: In the 1880s there was also a rise in anti-Chinese sentiment in the cities of Melbourne and Sydney. Earlier discontent had been curtailed by the segregationist policies in the rural protectorates and poorly reported in the urban publications. However, as more and more Chinese began moving from the country towns into the cities there was an equal rise in anti-Chinese sentiment. This resulted in another round of restrictive Acts in NSW in 1881 and 1888. It also contributed to a rising drive for Federation of Australia. One of the most compelling arguments for federation amongst the public and politicians of the time was that a united immigration policy would secure the borders of all the Australian colonies. The Chinese 'pest' or 'menace' was the root of these immigration fears. | END ID: 767

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 768 | TITLE: Dreamgirls (film) | CONTENT: For the 2007 Golden Globe Awards, Dreamgirls was nominated in five categories: Best Picture - Comedy or Musical, Best Actress in a Comedy or Musical (Beyoncé Knowles), Best Supporting Actor (Eddie Murphy), Best Supporting Actress (Jennifer Hudson), and Best Original Song ("Listen"). The film won the awards for Best Picture — Comedy or Musical, Best Supporting Actor, and Best Supporting Actress.[9] Dreamgirls received eight NAACP Image Award nominations, winning for Best Supporting Actress (Jennifer Hudson) and Outstanding Album (the soundtrack LP).[77] It was also named as one of the American Film Institute's top ten films of 2006. | END ID: 768

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 769 | TITLE: List of North Dakota State Bison in the NFL Draft | CONTENT: This is a list of North Dakota State Bison football players in the NFL Draft. | END ID: 769

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 770 | TITLE: Villanova Wildcats men's basketball | CONTENT: National Freshman of the Year | END ID: 770

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 771 | TITLE: Florentine Codex | CONTENT: SahagÃºn appeared to have asked questions about animals such as the following: | END ID: 771

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 772 | TITLE: Wangari Maathai | CONTENT: In 1977, Maathai founded the Green Belt Movement, an environmental non-governmental organization focused on the planting of trees, environmental conservation, and women's rights. In 1984, she was awarded the Right Livelihood Award, and in 2004, she became the first African woman to receive the Nobel Peace Prize for "her contribution to sustainable development, democracy and peace." Maathai was an elected member of Parliament and served as assistant minister for Environment and Natural resources in the government of President Mwai Kibaki between January 2003 and November 2005. She was an Honorary Councillor of the World Future Council. She made numerous achievements, was affiliated to many professional bodies and received several awards.[1] In 2011, Maathai died of complications from ovarian cancer. | END ID: 772

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 773 | TITLE: Maria Makiling | CONTENT: "From then on," Lanuza concludes, "Maria never let herself be seen by the people again. Every time somebody gets lost on the mountain, they remember the curse of the diwata. Yet they also remember the great love of Maria Makiling." | END ID: 773

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 774 | TITLE: Safe Drinking Water Act | CONTENT: In addition to requiring more contaminants to be regulated, the 1986 amendments included: | END ID: 774

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 775 | TITLE: Compulsory sterilization | CONTENT: Bangladesh is planning to introduce sterilization program in its overcrowded Rohingya refugee camps, where nearly a million refugees are fighting for space, after efforts to encourage birth control failed. Since 25 August 2017, more than 600,000 Rohingya Muslims have been fled from Rakhine state, Myanmar to neighboring Bangladesh, which is a Muslim majority country, following a military crackdown against Rohingya Muslims in Rakhine state, Myanmar. Sabura, a Rohingya mother of seven, said her husband believed the couple could support a large family. | END ID: 775

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 776 | TITLE: You Get Me (film) | CONTENT: When Tyler arrives, he sees Holly sitting in front of the fireplace, the first place he saw her the morning he woke up in her house previously. Holly tries recreating the weekend as Tyler runs around the house looking for Ali. He discovers Ali unconscious tied mid-air to the ceiling, forehead bleeding. He lowers her down, wakes her up, grabs a fire poker and starts trying to escape the house as Holly goes to get her gun. Tyler and Ali make it outside and before they get away, Holly stops them at gunpoint. Tyler tells Holly that he loves Ali and not her and he never will. Gil shows up behind them calling out Holly's name. Distracted, Holly shoots Tyler in the shoulder then attempts to shoot Gil but misses. Ali picks up the fire poker and stabs Holly in the side, causing Holly to fall back into the pool. Gil and Ali huddle around Tyler while waiting for the police to arrive. | END ID: 776

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 777 | TITLE: DNA replication | CONTENT: In a cell, DNA replication begins at specific locations, or origins of replication, in the genome.[3] Unwinding of DNA at the origin and synthesis of new strands, accommodated by an enzyme known as ligase, results in replication forks growing bi-directionally from the origin. A number of proteins are associated with the replication fork to help in the initiation and continuation of DNA synthesis. Most prominently, DNA polymerase synthesizes the new strands by adding nucleotides that complement each (template) strand. DNA replication occurs during the S-stage of interphase. | END ID: 777

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 778 | TITLE: Why Do Fools Fall in Love (album) | CONTENT: ^shipments figures based on certification alone | END ID: 778

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 779 | TITLE: Green tea | CONTENT: Green tea extract supplements are accessible over the counter in various forms. Standardized green tea extract is 90 percent total polyphenols, and 1 capsule equals 5 cups of tea.[6][7] | END ID: 779

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 780 | TITLE: Hawaiian religion | CONTENT: One Hawaiian creation myth is embodied in the Kumulipo, an epic chant linking the aliʻi, or Hawaiian royalty, to the gods.  The Kumulipo is divided into two sections: night, or pō, and day, or ao, with the former corresponding to divinity and the latter corresponding to mankind.  After the birth of Laʻilaʻi, the woman, and Kiʻi, the man, the man succeeds at seducing and reproducing with the woman before the god Kāne has a chance, thereby making the divine lineage of the gods younger than and thus subservient to the lineage of man.  This, in turn, illustrates the transition of mankind from being symbols for the gods (the literal meaning of kiʻi) into the keeper of these symbols in the form of idols and the like.[8] The Kumulipo was recited during the time of Makahiki, to honor the god of fertility, Lono.[9] | END ID: 780

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 781 | TITLE: Tampa Bay Rays | CONTENT: Pitchers
Starting rotation | END ID: 781

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 782 | TITLE: Five Nights at Freddy's: The Silver Eyes | CONTENT: On December 11, 2015, Scott Cawthon posted a teaser on his website for an upcoming untitled novel. According to him, the novel was written "alongside a professional writer for the last ten months" and "expands the mythos", revealing "a human element never before seen in the games". On December 15, 2015, Cawthon revealed the title of the book. The book was originally called Five Nights at Freddy's: The Untold Story, but was renamed shortly after.[2] It was supposed to be available for Amazon Kindle on December 22, 2015, but because of an error in Amazon's system, it was released slightly earlier on December 17, 2015. The paperback edition of the book was published by Scholastic on September 27, 2016.[3][4][5][6] | END ID: 782

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 783 | TITLE: Foster care | CONTENT: The use of expensive, brand name, patent protected medication was prevalent. In the case of SSRIs the use of the most expensive medications was noted to be 74%; in the general market only 28% are for brand name SSRI's vs generics. The average out-of-pocket expense per prescription was $34.75 for generics and $90.17 for branded products, a $55.42, difference.[65] | END ID: 783

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 784 | TITLE: Book series | CONTENT: Fictional series typically share a common setting, story arc, set of characters or timeline. They are common in genre fiction, particularly crime fiction, adventure fiction, and science fiction, as well as in children's literature. | END ID: 784

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 785 | TITLE: Trent Tucker Rule | CONTENT: On December 20, 2006, New York Knicks forward David Lee scored a game-winning basket with only 0.1 left on the clock. The shot counted because Lee deflected in the inbounds pass into the basket. This was the first occurrence of a team winning an NBA game with 0.1 left since Trent Tucker, and coincidentally from the same team, the New York Knicks.  Furthermore, this took place after the NBA adopted the Precision Time Systems unit, where officials, not the timer, start the clock.  In 2004, FIBA adopted a rule where the system would be mandatory in international competitions.[1] Michael Jordan, Charles Oakley and Patrick Ewing, all of whom participated in the original Trent Tucker game, were in attendance at the David Lee game in Madison Square Garden.[2] | END ID: 785

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 786 | TITLE: Women in Maya society | CONTENT: The status of women in Maya society can be inferred from their burials and textual and monumental history. Maya societies include Toniná, a city which developed a matrilineal system of hereditary descent after the reign and death of the powerful leader, Lady K’awil. She had assumed the mantle of power after the failure of the two male leaders.[2] Lady K'awil's reign is documented by murals that depict her seated on a throne with captives at her feet. | END ID: 786

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 787 | TITLE: The Pelican Brief (film) | CONTENT: The Pelican Brief has grossed $100 million in the United States and Canada, and $93 million in other territories, for a worldwide total of $195.3 million,[2][7][8] against a production budget of $45 million.[1] | END ID: 787

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 788 | TITLE: Ray Charles (musician, born 1918) | CONTENT: Discharged in 1946, Charles sang on New York radio ("Um Um Good" for Campbell's soups[10] among other gigs) and on many record dates. In 1947, he was the conductor for the Broadway hit Finian's Rainbow,[7] and conducted the original cast recording. Charles initially became associated with Perry Como in 1948 through his arrangements for the vocal group the Satisfiers. The group performed on Como's The Chesterfield Supper Club.[9][11] From 1949 to 1951, he was choral arranger-conductor on The Big Show, the last big radio variety show with Tallulah Bankhead and Meredith Willson.[12] Charles was also a soloist and sang in the choir on Manhattan Merry-Go-Round, Tuesday on Broadway, The Prudential Family Hour, The Celenese Hour, The Schafer Beer Program and The American Melody Hour, and he wrote the theme for Danny Kaye's 7-Up Radio Show. | END ID: 788

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 789 | TITLE: Where Eagles Dare | CONTENT: Vincent Canby of the New York Times gave a positive review, praising the action scenes and cinematography.[27] Likewise, Variety praised the movie, describing it as 'thrilling.'[28] The film was particularly lucrative for Richard Burton, who earned a considerable sum in royalties through television repeats and video sales.[29] Where Eagles Dare had its first showing on British television on 26 December 1979 on BBC1. In 2009 Cinema Retro magazine released a special issue dedicated to Where Eagles Dare which detailed the production and filming of the movie.[30] Today, Where Eagles Dare is considered one of the best war films of all time.[31] | END ID: 789

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 790 | TITLE: Andy Murray | CONTENT: Speaking at a press conference after the match, Murray said, "As it is, I'd be very surprised if I was playing in Paris. I need to make a plan as to what I do. I'll chat with the guys tonight and make a plan for the next few days then make a decision on Paris after the next five days."[212] He would go on to withdraw from Roland Garros later, citing a back injury.[213] After a four-week break due to injury, Murray made his comeback at the 2013 Aegon Championships, where he was the top seed. After a rain delayed first day, Murray had to complete his second round match against Nicolas Mahut, and his subsequent match against Marinko Matosevic on the same day, both of which he won in straight sets. After beating Benjamin Becker in the quarter-finals, Murray next faced his first top ten opponent since losing to Tomáš Berdych in Madrid, taking on Jo-Wilfried Tsonga in the semi-finals. After dropping the first set against the Frenchman, Murray eventually raised his level and won in three to set up a final against Marin Čilić of Croatia, his third consecutive final on grass courts. He came from behind again to beat Čilić in three sets to claim his third title at Queen's Club.[214] | END ID: 790

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 791 | TITLE: Zürich | CONTENT: The previous boundaries of the city of Zürich (before 1893) were more or less synonymous with the location of the old town. Two large expansions of the city limits occurred in 1893 and in 1934 when the city of Zürich merged with many surrounding municipalities, that had been growing increasingly together since the 19th century. Today, the city is divided into twelve districts (known as Kreis in German), numbered 1 to 12, each one of which contains between one and four neighborhoods: | END ID: 791

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 792 | TITLE: South-Western City School District (Franklin County, Ohio) | CONTENT: The South-Western City School District is Ohio's sixth largest public school district located southwest of the city of Columbus. The district serves nearly 20,000 students throughout the southwest quadrant of Franklin County, including the cities of Galloway, Georgesville, Grove City, and Urbancrest. The district also serves all of Franklin, Jackson, Pleasant, and Prairie townships and a portion of Columbus. | END ID: 792

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 793 | TITLE: Inflation in India | CONTENT: It arises as the basis theme in deciding an adequate monetary policy. There are two debatable proportions for an effective inflation, whether it should be in the range of 1–3 per cent as the inflation rate that persists in the industrialized economy or should it be in the range of 6–7 per cent. While deciding on the elaborate inflation rate certain problems occur regarding its measurement. The measurement bias has often calculated an inflation rate that is comparatively more than in nature. Secondly, there often arises a problem when the quality improvements in the product are in need to be captured out, hence it affects the price index. The consumer preference for a cheaper goods affects the consumption basket at costs, for the increased expenditure on the cheaper goods takes time for the increased weight and measuring inflation. The Boskin Commission has measured 1.1 per cent of the increased inflation in USA every annum. The commission points out for the developed countries comprehensive study on inflation to be fairly low. | END ID: 793

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 794 | TITLE: Font | CONTENT: Alternative characters are often called stylistic alternates. These may be switched on to allow users more flexibility to customise the font to suit their needs. The practice is not new: in the 1930s, Gill Sans, a British design, was sold abroad with alternative characters to make it resemble fonts such as Futura popular in other countries, while Bembo from the same period has two styles of 'R': one with a stretched-out leg, matching its fifteenth-century model, and one less-common shorter version.[29] With modern digital fonts, it is possible to group related alternative characters into stylistic sets, which may be turned on and off together. For example, in Williams Caslon Text, a revival of the 18th century font Caslon, the default italic forms have many swashes matching the original design. For a more spare appearance, these can all be turned off at once by engaging stylistic set 4.[30] Junicode, intended for academic publishing, uses ss15 to enable a variant form of 'e' used in medieval Latin. A corporation commissioning a modified version of a commercial font for their own use, meanwhile, might request that their preferred alternates be set to default. | END ID: 794

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 795 | TITLE: West Coast jazz | CONTENT: Although West Coast jazz is often compared to the cool style, Los Angeles musicians locally known as "hard swingers," "blew bop as tough as anything emerging out of Detroit and New York...."[1] In later years, their music was known as "California Hard." Roy Carr notes that this is not surprising. By the late 1940s, the Central Avenue scene had the most bebop musicians outside of New York. Max Roach and Clifford Brown, Shelly Manne, and Curtis Counce established harder-sounding bands in Los Angeles.[1] | END ID: 795

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 796 | TITLE: Dome | CONTENT: The new building materials of the 19th century and a better understanding of the forces within structures from the 20th century has opened up new possibilities. Iron and steel beams, steel cables, and pre-stressed concrete have eliminated the need for external buttressing and enabled far thinner domes. Whereas earlier masonry domes may have had a radius to thickness ratio of 50, the ratio for modern domes can be in excess of 800. The lighter weight of these domes has not only permitted far greater spans, but also allowed for the creation of large movable domes over modern sports stadiums.[38] | END ID: 796

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 797 | TITLE: Expectation–maximization algorithm | CONTENT: EM is frequently used for data clustering in machine learning and computer vision. In natural language processing, two prominent instances of the algorithm are the Baum-Welch algorithm for hidden Markov models, and the inside-outside algorithm for unsupervised induction of probabilistic context-free grammars. | END ID: 797

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 798 | TITLE: John Cooper Clarke | CONTENT: His first job was a laboratory technician at Salford Tech.[6] He began his performance career in Manchester folk clubs, where he began working with Rick Goldstraw and his band the Ferrets.[2] His first releases were on Tosh Ryan and Martin Hannett's independent label Rabid,[7] starting with the EP Innocents in October 1977.[2] Rabid also released his debut LP OÃ¹ est la maison de fromage'? (catalogue number NOZE 1), which was a collection of live recordings, demos and rehearsals. This was reissued by Revolver Records in 1989 (RRLP 10) also making it his last album to date. He toured with Bill Nelson's band Be-Bop Deluxe in 1978 and was signed by Epic Records, who issued the Martin Hannett produced studio album Disguise In Love in 1978.[2] | END ID: 798

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 799 | TITLE: Real ID Act | CONTENT: Another privacy concern raised by privacy advocates such as the Electronic Frontier Foundation is that the implementation of the Real ID Act will make it substantially easier for the government to track numerous activities of Americans and conduct surveillance.[118][119] | END ID: 799

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 800 | TITLE: ABC Board | CONTENT: During their 2007 federal election campaign, Labor announced plans to introduce a new system, similar to that of the BBC, for appointing members to the board.[23][24] Under this new system, now in place, ABC candidates are considered by a panel established "at arm's length" from the Communications Minister.[25] If the Minister chose someone not on the panel's shortlist, the Minister would be required to justify their selection to Australian Parliament. The Chairman of the ABC is nominated by the Prime Minister and endorsed by the Leader of the Opposition.[23][26] | END ID: 800

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 801 | TITLE: Forest of Dean | CONTENT: The Forest of Dean is known for its birds; pied flycatchers, redstarts, wood warblers and hawfinches can be seen at RSPB Nagshead. The mixed forest supports Britain's best concentration[citation needed] of goshawks and a viewing site at New Fancy is manned during February and March. Peregrine falcons can be seen from the viewpoint at Symonds Yat rock. Mandarin ducks, which nest in the trees, and reed warblers can be seen at Cannop Ponds and Cannop Brook, running from the ponds through Parkend, is famed for its dippers. | END ID: 801

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 802 | TITLE: Rainbow flag | CONTENT: Flag of Cusco, Peru | END ID: 802

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 803 | TITLE: Maneki-neko | CONTENT: The Pokémon named Meowth is based upon the maneki-neko.[6] | END ID: 803

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 804 | TITLE: Parable of the Workers in the Vineyard | CONTENT: In Matthew Matt 20:1–16, Jesus says that any "laborer" who accepts the invitation to the work in the vineyard (said by Jesus to represent the Kingdom of Heaven), no matter how late in the day, will receive an equal reward with those who have been faithful the longest. | END ID: 804

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 805 | TITLE: Shripad Amrit Dange | CONTENT: Dange was later elected to the 4th Lok Sabha in 1967 from Bombay City (Central) Constituency of the Maharashtra State.[28] | END ID: 805

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 806 | TITLE: Seventh Son of a Seventh Son | CONTENT: Seventh Son of a Seventh Son is the seventh studio album by English heavy metal band Iron Maiden, released on 11 April 1988 by the EMI label in Europe, and its sister label Capitol in North America. It was re-released on 2002 by Sanctuary/Columbia in the United States. | END ID: 806

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 807 | TITLE: Economy of the Ming dynasty | CONTENT: The economy of the Ming dynasty (1368â€“1644) of China was the largest in the world during that period. It is regarded as one of China's three golden ages (the other two being the Han and Song periods). The period was marked by the increasing political influence of the merchants, the gradual weakening of imperial rule, and technological advances. | END ID: 807

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 808 | TITLE: Christmas tree | CONTENT: Although the tradition of decorating the home with evergreens was long established,[32] the custom of decorating an entire small tree was unknown in Britain until some two centuries ago. At the time of the personal union with Hanover, George III's German-born wife, Charlotte of Mecklenburg-Strelitz, introduced a Christmas tree at a party she gave for children in 1800.[33] The custom did not at first spread much beyond the royal family.[34] Queen Victoria as a child was familiar with it and a tree was placed in her room every Christmas. In her journal for Christmas Eve 1832, the delighted 13-year-old princess wrote:[35] | END ID: 808

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 809 | TITLE: Jawaharlal Nehru Centre for Advanced Scientific Research | CONTENT: The Jawaharlal Nehru Centre for Advanced Scientific Research (JNCASR) is a multidisciplinary research institute located at Jakkur, Bangalore, India. It was established by the Department of Science and Technology of the Government of India, to mark the birth centenary of Pandit Jawaharlal Nehru. | END ID: 809

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 810 | TITLE: Barack Obama "Hope" poster | CONTENT: After the initial 700 posters, the Obama campaign conveyed through Sergant that they wanted to promote the theme of hope, and most of the posters sold by Fairey subsequently had the word "hope" and later "change" instead of "progress"; the obey star was also absent from later versions. By October 2008, Fairey and Sergant claimed to have printed 300,000 posters (with less than 2,000 sold and the rest given away or displayed) and 1,000,000 stickers, as well as clothing and other items with the image sold through Fairey's website, in addition to copies printed by others.[12][14]  According to Fairey and Sergant, proceeds from sales of the image were used to produce more posters and other merchandise in support of the Obama campaign, rather than direct profit for Fairey.[12] | END ID: 810

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 811 | TITLE: David Price (baseball) | CONTENT: In 2006, Price posted a 9–5 record with a 4.16 ERA in 110 1⁄3 innings pitched. He set a school single-season record in strikeouts with 155 while walking only 43 batters. Over a span of six starts early in the season, he recorded 10 or more strikeouts each game, including a 17-strikeout performance in a game against Arkansas.[2] That year, he was one of five finalists for the Golden Spikes Award and a semifinalist for the Roger Clemens Award. He was also named to the third-team All-American by the National Collegiate Baseball Writers Association, first-team All-South Region by the American Baseball Coaches Association and second-team All-SEC by the coaches in that conference.[2] | END ID: 811

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 812 | TITLE: X Games | CONTENT: 25th medal, landing the 1st switch ollie 540. | END ID: 812

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 813 | TITLE: Mehran Karimi Nasseri | CONTENT: Having claimed to have one British parent, although he has produced no evidence to support this claim, he decided to settle in the UK in 1986, but en route there in 1988, his papers were lost when his briefcase was allegedly stolen.[4] (Others indicate that Nasseri actually mailed his documents to Brussels while onboard a ferry to Britain, lying about them being stolen.[5]) Despite this setback, he boarded the plane for London but was promptly returned to France when he failed to present a passport to British immigration officials. He was initially arrested by the French, but then released as his entry to the airport was legal and he had no country of origin to be returned to; thus began his residency at Terminal 1. | END ID: 813

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 814 | TITLE: Doctor Who missing episodes | CONTENT: Unrelated to the regular archive purges, the final shot of The Deadly Assassin Episode 3 (1976) has been excised from the master copy. The shot was removed after its initial UK transmission, following complaints from Mary Whitehouse of the National Viewers' and Listeners' Association.[12] Subsequent repeats and commercial releases have restored the shot from off-air video copies.[12] | END ID: 814

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 815 | TITLE: Ultraviolet index | CONTENT: The ultraviolet index or UV Index is an international standard measurement of the strength of sunburn-producing ultraviolet (UV) radiation at a particular place and time. The scale was developed by Canadian scientists in 1992, then adopted and standardized by the UN's World Health Organization and World Meteorological Organization in 1994. It is primarily used in daily forecasts aimed at the general public, and is increasingly available as an hourly forecast as well. | END ID: 815

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 816 | TITLE: Keep Talking and Nobody Explodes | CONTENT: Destructoid awarded it a score of 9 out of 10, saying "If you are tired of always playing Cards Against Humanity, Monopoly, and that Gargoyles board game on Laserdisc, then Keep Talking and Nobody Explodes will certainly give you the fix you're looking for, pending you have friends ready to be committed to the task at hand. If not, Gargoyles is always a good choice."[14] | END ID: 816

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 817 | TITLE: Rocky Mountain National Park | CONTENT: Region 4 is the heart of the park with easy road and trail access, great views, and lake hikes including the most popular trails.[67] Flattop Mountain is a tundra hike and the easiest hike to the Continental Divide. The hike up Hallett Peak passes through three climate zones, crossing over Flattop Mountain, traversing the ridge that supports Tyndall Glacier, and finally ascending to the summit of Hallett Peak.[76] | END ID: 817

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 818 | TITLE: Soul2Soul II Tour | CONTENT: Hill/McGraw | END ID: 818

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 819 | TITLE: Ship | CONTENT: Following the Exxon Valdez spill, the United States passed the Oil Pollution Act of 1990 (OPA-90), which included a stipulation that all tankers entering its waters be double-hulled by 2015. Following the sinkings of Erika (1999) and Prestige (2002), the European Union passed its own stringent anti-pollution packages (known as Erika I, II, and III), which require all tankers entering its waters to be double-hulled by 2010. The Erika packages are controversial because they introduced the new legal concept of "serious negligence".[80] | END ID: 819

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 820 | TITLE: Pollination | CONTENT: Peaches are considered self-fertile because a commercial crop can be produced without cross-pollination, though cross-pollination usually gives a better crop. Apples are considered self-incompatible, because a commercial crop must be cross-pollinated. Many commercial fruit tree varieties are grafted clones, genetically identical. An orchard block of apples of one variety is genetically a single plant. Many growers now consider this a mistake. One means of correcting this mistake is to graft a limb of an appropriate pollenizer (generally a variety of crabapple) every six trees or so.[citation needed] | END ID: 820

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 821 | TITLE: Quadratic equation | CONTENT: A quadratic equation with real or complex coefficients has two solutions, called roots. These two solutions may or may not be distinct, and they may or may not be real. | END ID: 821

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 822 | TITLE: Anthracite | CONTENT: Geologically, the largest most concentrated anthracite deposit in the world is found in northeastern Pennsylvania, United States. Locally called the Coal Region, the deposit contains 480 square miles of coal bearing rock which originally held 22.8 billion short tons (20.68 billion tonnes) of anthracite.[24] (The geographic region is roughly 100 miles (161 km) in length and 30 miles (48 km) in width.) Because of historical mining and development of the lands overlying the coal, it is estimated that 7 billion short tons (6.3 billion tonnes) of minable reserves remain. The United States also contains several smaller deposits of anthracite, such as those historically mined in Crested Butte, Colorado. | END ID: 822

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 823 | TITLE: Counties of Ireland | CONTENT: Parts of some towns and cities were exempt from the jurisdiction of the counties that surrounded them. These towns and cities had the status of a County corporate, many granted by Royal Charter, which had all the judicial, administrative and revenue raising powers of the regular counties. | END ID: 823

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 824 | TITLE: Modernity | CONTENT: Modernist republicanism openly influenced the foundation of republics during the Dutch Revolt (1568–1609) (Bock, Skinner, and Viroli 1990, chapt. 10,12[page needed]), English Civil War (1642–1651) (Rahe 2006, chapt. 1[page needed]), American Revolution (1775–1783) (Rahe 2006, chapt. 6–11[page needed]), the French Revolution (1789–1799), and the Haitian revolution (1791–1804). (Orwin and Tarcov 1997, chapt. 8[page needed]). | END ID: 824

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 825 | TITLE: Louis XIV of France | CONTENT: In 1910, the American historical novelist Charles Major wrote "The Little King: A Story of the Childhood of King Louis XIV". | END ID: 825

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 826 | TITLE: Nashville (2012 TV series) | CONTENT: The Music of Nashville: Season 5, Volume 1 was released on March 10, 2017. | END ID: 826

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 827 | TITLE: Colby cheese | CONTENT: In 1885, Joseph F. Steinwand developed a new type of cheese at his father's cheese factory near Colby, Wisconsin. The cheese was named after the village,[1] which had been founded three years earlier.[2] While Colby cheese is still widely available, it is no longer produced in Colby. A festival commemorating the cheese is held every year in mid-July where all local food booths offer free Colby cheese. On August 12, 2015, the original cheese factory was torn down leaving only the foundations of the building.[citation needed] | END ID: 827

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 828 | TITLE: Law | CONTENT: The Classical republican concept of "civil society" dates back to Hobbes and Locke.[128] Locke saw civil society as people who have "a common established law and judicature to appeal to, with authority to decide controversies between them."[129] German philosopher Georg Wilhelm Friedrich Hegel distinguished the "state" from "civil society" (bÃ¼rgerliche Gesellschaft) in Elements of the Philosophy of Right.[130] | END ID: 828

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 829 | TITLE: Brazil at the 2010 FIFA World Cup | CONTENT: Assistant referees:
Héctor Vergara (Canada)
Marvin Torrentera (Mexico)
Fourth official:
Peter O'Leary (New Zealand)
Fifth official:
Brent Best (New Zealand) | END ID: 829

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 830 | TITLE: Bill of Rights 1689 | CONTENT: The Bill of Rights, also known as the English Bill of Rights, is an Act of the Parliament of England that deals with constitutional matters and sets out certain basic civil rights. It received the Royal Assent on 16 December 1689 and is a restatement in statutory form of the Declaration of Right presented by the Convention Parliament to William III and Mary II in February 1689, inviting them to become joint sovereigns of England. The Bill of Rights lays down limits on the powers of the monarch and sets out the rights of Parliament, including the requirement for regular parliaments, free elections, and freedom of speech in Parliament. It sets out certain rights of individuals including the prohibition of cruel and unusual punishment and reestablished Protestants to have arms for their defence within the rule of law. Furthermore, the Bill of Rights described and condemned several misdeeds of James II of England.[1] | END ID: 830

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 831 | TITLE: Official languages of the United Nations | CONTENT: The official languages of the United Nations are the six languages that are used in UN meetings, and in which all official UN documents are written. In alphabetical order, they are: | END ID: 831

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 832 | TITLE: Languages of Sweden | CONTENT: The Kingdom of Sweden is a nation-state for the Swedish people,[citation needed] and as such, their national language is held in very high regard. Of Sweden's roughly ten million people, almost all speak Swedish as at least a second language, and the majority as a first language (7,825,000, according to SIL's Ethnologue). Swedish is also an official language in Finland where it is spoken by a large number of Swedish-speaking Finns. The language is also spoken to some degree by ethnic Swedes living outside Sweden, for example, just over half a million people of Swedish descent in the United States speak the language, according to Ethnologue.[citation needed] | END ID: 832

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 833 | TITLE: Card counting | CONTENT: Back-counting is generally done on shoe games, of 4, 6, or 8 decks, although it can be done on pitch games of 1 or 2 decks. The reason for this is that the count is more stable in a shoe game, so a player will be less likely to sit down for one or two hands and then have to get up. In addition, many casinos do not allow "mid-shoe entry" in single or double deck games which makes Wonging impossible. Another reason is that many casinos exhibit more effort to thwart card counters on their pitch games than on their shoe games, as a counter has a smaller advantage on an average shoe game than in a pitch game.[12] | END ID: 833

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 834 | TITLE: United States Army Rangers | CONTENT: In 1966, a panel headed by General Ralph E. Haines, Jr. recommended making Ranger training mandatory for all Regular Army officers upon commissioning. "On 16 August 1966, the Chief of Staff of the Army, General Harold K. Johnson, directed it so." This policy was implemented in July 1967. It was rescinded on 21 June 1972 by General William Westmoreland. Once again, Ranger training was voluntary.[30]:28â€“29 In August 1987, the Ranger Department was split from the Infantry School and the Ranger Training Brigade was established, commanded by Brigadier General (R) James Emory Mace. | END ID: 834

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 835 | TITLE: g factor (psychometrics) | CONTENT: Some researchers believe that there is a threshold level of g below which socially significant creativity is rare, but that otherwise there is no relationship between the two. It has been suggested that this threshold is at least one standard deviation above the population mean. Above the threshold, personality differences are believed to be important determinants of individual variation in creativity.[118][119] | END ID: 835

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 836 | TITLE: University of the Philippines College of Medicine | CONTENT: Due to the program's two entry points, Direct Entrants are joined by the Lateral Entrants as both groups enter LU III. This results in one class of about 160 students in medicine proper. | END ID: 836

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 837 | TITLE: Briggs & Stratton | CONTENT: Eventually Briggs and Stratton settled on manufacturing automotive components and small gasoline engines. Briggs purchased an engine patent from A.O. Smith Company and began powering early washing machines and reel mowers as well as many other types of equipment. The company went public on the New York Stock Exchange in 1928. | END ID: 837

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 838 | TITLE: It's a Mad, Mad, Mad, Mad World | CONTENT: Claims of attempts to produce a sequel to It's a Mad, Mad, Mad, Mad World have circulated, but no film producer has officially confirmed a sequel contract despite multiple  attempts.[35][36] | END ID: 838

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 839 | TITLE: Edward VI of England | CONTENT: After 1551, the Reformation advanced further, with the approval and encouragement of Edward, who began to exert more personal influence in his role as Supreme Head of the church.[132] The new changes were also a response to criticism from such reformers as John Hooper, Bishop of Gloucester, and the Scot John Knox, who was employed as a minister in Newcastle upon Tyne under the Duke of Northumberland and whose preaching at court prompted the king to oppose kneeling at communion.[133] Cranmer was also influenced by the views of the continental reformer Martin Bucer, who died in England in 1551, by Peter Martyr, who was teaching at Oxford, and by other foreign theologians.[134] The progress of the Reformation was further speeded by the consecration of more reformers as bishops.[135] In the winter of 1551â€“52, Cranmer rewrote the Book of Common Prayer in less ambiguous reformist terms, revised canon law, and prepared a doctrinal statement, the Forty-two Articles, to clarify the practice of the reformed religion, particularly in the divisive matter of the communion service.[136] Cranmer's formulation of the reformed religion, finally divesting the communion service of any notion of the real presence of God in the bread and the wine, effectively abolished the mass.[137] According to Elton, the publication of Cranmer's revised prayer book in 1552, supported by a second Act of Uniformity, "marked the arrival of the English Church at Protestantism".[138] The prayer book of 1552 remains the foundation of the Church of England's services.[139] However, Cranmer was unable to implement all these reforms once it became clear in spring 1553 that King Edward, upon whom the whole Reformation in England depended, was dying.[140] | END ID: 839

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 840 | TITLE: Rani Rashmoni | CONTENT: Presently, the Lokamata Rani Rashmoni Mission is situated at Nimpith, South 24 Parganas, West Bengal, 743338, India.[2] | END ID: 840

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 841 | TITLE: Castleford | CONTENT: Castleford is a town in the metropolitan borough of Wakefield, West Yorkshire, England. It had a population of 40,210 at the 2011 Census.[1][2][3] Historically in the West Riding of Yorkshire, to the north of the town centre the River Calder joins the River Aire and the Aire and Calder Navigation. | END ID: 841

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 842 | TITLE: Baltimore | CONTENT: The city is named after Cecil Calvert, second Lord Baltimore,[19] (1605–1675),[20] of the Irish House of Lords and founding proprietor of the Province of Maryland.[21][22] Baltimore Manor was the name of the estate in County Longford on which the Calvert family lived in Ireland.[22][23] Baltimore is an anglicization of the Irish name Baile an Tí Mhóir, meaning "town of the big house."[22] | END ID: 842

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 843 | TITLE: Direct Action Day | CONTENT: The political rivalry among the major nationalist parties in Bengal took a form different from that in New Delhi, mainly because of the broad mass base those organizations enjoyed and the tradition of flexible political dealing in which they excelled. At the initial stage of the riots, the Congress and the Muslim League appeared to be confident that they could draw on this tradition even if a difficult situation arose out of political showdown. Most probably, Direct Action Day in Calcutta was planned to be a large-scale hartal and mass rally (which is an accepted part of political culture in Calcutta) which they knew very well how to control. However, the response from the masses far exceeded any expectations. The political leaders seriously miscalculated the strong emotional response that the word 'nation', as interpreted under the new situation, had evoked. In August 1946 the 'nation' was no longer a mere political slogan. It was rapidly turning into 'reality' both in realpolitik and in people's imaginations. The system to which Bengal political leaders had grown accustomed for decades could not cope with this dynamic change. As we have seen, it quickly and easily broke down on the first day of the disturbances.[7] | END ID: 843

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 844 | TITLE: Latin America–United States relations | CONTENT: The U.S. sent troops to the border with Mexico when it became clear in March 1911 that the regime of Porfirio Díaz could not control revolutionary violence.[33] Díaz resigned, opening the way for free elections that brought Francisco I. Madero to the presidency in November 1911. The U.S. Ambassador to Mexico, Henry Lane Wilson, conspired with opposition forces to topple Madero's regime in February 1913, during what is known as the Ten Tragic Days. | END ID: 844

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 845 | TITLE: Early modern warfare | CONTENT: Column - This formation was typically used while marching, although with sufficient will and mass it was effective at breaking through line formations, albeit with heavy casualties. | END ID: 845

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 846 | TITLE: Joseph Stalin | CONTENT: Stalin had a soft voice,[721] and when speaking Russian he did so slowly, carefully choosing his phrasing.[711] Although he avoided doing so in public, in private Stalin used coarse language.[722] Described as a poor orator,[723] according to Volkogonov, Stalin's speaking style was "simple and clear, without flights of fancy, catchy phrases or platform histrionics".[724] He rarely spoke before large audiences, and preferred to express himself in written form.[725] His writing style was similar, being characterised by its simplicity, clarity, and conciseness.[726] | END ID: 846

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 847 | TITLE: Future of space exploration | CONTENT: In terms of propulsion, the main challenge is the liftoff and initial momentum, since there is no friction in the vacuum of space. Based on the missions goals, including factors such as distance, load and time of flight, the type of propulsion drive used, planned to use, or in design varies from chemical propellants, such as liquid hydrogen and oxidizer[16] (Space Shuttle Main Engine), to plasma[15] or even nanoparticle propellants.[17] | END ID: 847

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 848 | TITLE: Dab (dance) | CONTENT: In Saudi Arabia, the move was made illegal by the National Committee for Drug Control as it was deemed that it "alludes to weed and other illegal substances." In August 2017, Saudi singer and actor Abdallah Al Shaharani was arrested for performing the move at a music festival in Ta'if.[22] | END ID: 848

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 849 | TITLE: Matthew 7:2 | CONTENT: As Schweizer notes this verse, if read literally, is a contradiction of the previous one. While the first says not to judge, this one established rules for judging.[1] Luz advances the explanation that this verse states that if you search to find faults with others, that God will then search to find fault with you, and since all humans are infinitely flawed you would then easily be condemned. Thus even a small amount of judging by a person will bring a great punishment form God, and this verse essentially repeats the argument of the first against judging. More scholars simply believe that the condemnation of judging in Matthew 7:1 is far from absolute.[2] | END ID: 849

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 850 | TITLE: Software bug | CONTENT: Bug management includes the process of documenting, categorizing, assigning, reproducing, correcting and releasing the corrected code. Proposed changes to software – bugs as well as enhancement requests and even entire releases – are commonly tracked and managed using bug tracking systems or issue tracking systems. The items added may be called defects, tickets, issues, or, following the agile development paradigm, stories and epics. Categories may be objective, subjective or a combination, such as version number, area of the software, severity and priority, as well as what type of issue it is, such as a feature request or a bug. | END ID: 850

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 851 | TITLE: United States Army Corps of Engineers | CONTENT: Review of Corps of Engineers' projects has also been criticized for its lack of impartiality. The investigation of levee failure in New Orleans during Hurricane Katrina was sponsored by the American Society of Civil Engineers (ASCE) but funded by the Corps of Engineers and involved its employees.[42][43] | END ID: 851

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 852 | TITLE: French franc | CONTENT: Coins were freely exchangeable until 17 February 2005 at Banque de France only (some commercial banks also accepted the old coins but were not required to do so for free after the transition period in 2001), by converting their total value in francs to euros (rounded to the nearest cent) at the fixed rate of 6.55957 francs for 1 euro. Banknotes remained convertible up until 17 February 2012.[22] By that date, franc notes worth some â‚¬550 million remained unexchanged, allowing the French state to register the corresponding sum as revenue.[23] | END ID: 852

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 853 | TITLE: Night on Bald Mountain | CONTENT: The following scenario is taken from Rimsky-Korsakov's later "magic opera-ballet" Mlada (1890), based on the same libretto by Viktor KrÃ¯lov. | END ID: 853

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 854 | TITLE: Respiratory failure | CONTENT: Type 1 respiratory failure is defined as a low level of oxygen in the blood (hypoxemia) without an increased level of carbon dioxide in the blood (hypercapnia), and indeed the PaCO2 may be normal or low. It is typically caused by a ventilation/perfusion (V/Q) mismatch; the volume of air flowing in and out of the lungs is not matched with the flow of blood to the lungs. The basic defect in type 1 respiratory failure is failure of oxygenation characterized by: | END ID: 854

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 855 | TITLE: The Star-Spangled Banner | CONTENT: When the National Anthem was first recognized by law in 1932, there was no prescription as to behavior during its playing. On June 22, 1942, the law was revised indicating that those in uniform should salute during its playing, while others should simply stand at attention, men removing their hats. (The same code also required that women should place their hands over their hearts when the flag is displayed during the playing of the Anthem, but not if the flag was not present.) On December 23, 1942 the law was again revised instructing men and women to stand at attention and face in the direction of the music when it was played. That revision also directed men and women to place their hands over their hearts only if the flag was displayed. Those in uniform were required to salute. On July 7, 1976, the law was simplified. Men and women were instructed to stand with their hands over their hearts, men removing their hats, irrespective of whether or not the flag was displayed and those in uniform saluting. On August 12, 1998, the law was rewritten keeping the same instructions, but differentiating between "those in uniform" and "members of the Armed Forces and veterans" who were both instructed to salute during the playing whether or not the flag was displayed. Because of the changes in law over the years and confusion between instructions for the Pledge of Allegence versus the National Anthem, throughout most of the 20th century many people simply stood at attention or with their hands folded in front of them during the playing of the Anthem, and when reciting the Pledge they would hold their hand (or hat) over their heart. After 9/11, the custom of placing the hand over the heart during the playing of the Anthem became nearly universal. [57][58][59] | END ID: 855

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 856 | TITLE: Identity document | CONTENT: Sri Lanka is in the process of developing a Smart Card based RFID NIC card which will replace the obsolete 'laminated type' cards by storing the holders information on a chip that can be read by banks, offices, etc., thereby reducing the need to have documentation of these data physically by storing in the cloud. | END ID: 856

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 857 | TITLE: Wikipedia:Autobiography | CONTENT: Writing an autobiography on Wikipedia is an example of conflict of interest editing and is strongly discouraged. Editing a biography about yourself is acceptable only if you are removing unambiguous vandalism or clear-cut and serious violations of our biography of living persons policy. | END ID: 857

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 858 | TITLE: Dirty Dancing (2017 film) | CONTENT: Filming was based in Hendersonville, North Carolina. Most of the filming locations were across western North Carolina including Asheville, Cashiers and Saluda, with filming taking place in April and May 2016.[18][19] People living in the Hendersonville area served as crew members, extras and dancers, and they were invited to provide cars from the 1960s. Much of the filming took place at High Hampton Inn in Cashiers.[20] It created an estimated 1,225 jobs, including 900 extras, 30 cast members and 225 crew positions to support the project.[21] | END ID: 858

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 859 | TITLE: Fire appliances in the United Kingdom | CONTENT: New Dimension vehicles are large curtain-side trucks designed to be deployed at incidents involving CBRN materials or for urban search and rescue (USAR) use at the scenes of natural or large-scale disasters. | END ID: 859

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 860 | TITLE: Economy of Egypt | CONTENT: The transition to the unified exchange rate regime was completed in December 2004. Shortly later, Egypt has notified the International Monetary Fund (IMF) that it has accepted the obligations of Article VIII, Section 2, 3, and 4 of the IMF Articles of Agreement, with effect from 2 January 2005. IMF members accepting the obligations of Article VIII undertake to refrain from imposing restrictions on the making of payments and transfers for current international transactions, or from engaging in discriminatory currency arrangements or multiple currency practices, except with IMF approval. | END ID: 860

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 861 | TITLE: House of the Temple | CONTENT: Designed by John Russell Pope, it stands at 1733 16th Street, N.W., in the Dupont Circle neighborhood, about one mile directly north of the White House.  The full name of the Supreme Council is "The Supreme Council (Mother Council of the World) of the Inspectors General Knights Commander of the House of the Temple of Solomon of the Thirty-third degree of the Ancient and Accepted Scottish Rite of Freemasonry of the Southern Jurisdiction of the United States of America." | END ID: 861

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 862 | TITLE: L. S. Lowry | CONTENT: Five Lowry art works were stolen from the Grove Fine Art Gallery in Cheadle Hulme, Stockport on 2 May 2007. The most valuable were The Viaduct, estimated value of £700,000 and The Tanker Entering the Tyne, which is valued at over £500,000. The Surgery, The Bridge at Ringley and The Street Market were also stolen.[81] The paintings were later found in a house in Halewood near Liverpool.[82] | END ID: 862

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 863 | TITLE: Once Upon a Time (TV series) | CONTENT: There are also worlds known as the Realms of Storytelling.[90] These realms are mostly based on the Land Without Magic, taking the name of a certain location and a certain time period. Known worlds are the Land Without Color,[83] 19th Century London,[91] Victorian England,[92] Kansas,[93] 1920s England,[94] 19th Century France,[95] and the Land Without Stories.[96] | END ID: 863

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 864 | TITLE: El Paso–Juárez | CONTENT: In the later 19th century the population in the region began to grow rapidly. With the arrival of the Southern Pacific, Texas and Pacific and the Atchison, Topeka and Santa Fe railroads in 1881, trade with the rest of the U.S. increased substantially. The area attracted newcomers ranging from businessmen and priests, to gunfighters and prostitutes. In the U.S. El Paso became known as the "Six Shooter Capital" because of its lawlessness.[16] Prostitution and gambling flourished. During World War I, the U.S. Department of the Army pressured El Paso authorities to crack down on vice, creating a tourist boom in JuÃ¡rez whose vice businesses continued to thrive. | END ID: 864

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 865 | TITLE: Blue | CONTENT: Man's suit, 1826. Dark blue suits were still rare; this one is blue-green or teal. | END ID: 865

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 866 | TITLE: Genome size | CONTENT: Obligate endosymbiotic species are characterized by a complete inability to survive external to their host environment. These species have become a considerable threat to human health, as they are often highly capable of evading human immune systems and manipulating the host environment to acquire nutrients. A common explanation for these keen manipulative abilities is the compact and efficient genomic structure consistently found in obligate endosymbionts. This compact genome structure is the result of massive losses of extraneous DNA - an occurrence that is exclusively associated with the loss of a free-living stage. In fact, as much as 90% of the genetic material can be lost when a species makes the evolutionary transition from a free-living to obligate intracellular lifestyle. Common examples of species with reduced genomes include: Buchnera aphidicola, Rickettsia prowazekii and Mycobacterium leprae. One obligate endosymbiont of leafhoppers, Nasuia deltocephalinicola, has the smallest genome currently known among cellular organisms at 112kb.[15] It is important to note, however, that some obligate intracellular species have positive fitness effects on their hosts. (See also mutualists and parasites.) | END ID: 866

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 867 | TITLE: B. R. Ambedkar | CONTENT: Bhimrao Ramji Ambedkar (14 April 1891 – 6 December 1956), popularly known as Baba Saheb, was an Indian jurist, economist, politician and social reformer who inspired the Dalit Buddhist Movement and campaigned against social discrimination against Untouchables (Dalits), while also supporting the rights of women and labour.[3][4] He was Independent India's first law minister, the principal architect of the Constitution of India and a founding father of the Republic of India.[5][6][7][8][9] | END ID: 867

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 868 | TITLE: Super Street Fighter II Turbo HD Remix | CONTENT: Super Street Fighter II Turbo HD Remix is a two dimensional fighting game released using the PlayStation Store and Xbox Live Arcade download services. A physical copy of the game was later released as part of Capcom Digital Collection. It is a remake of Super Street Fighter II Turbo (the fifth arcade iteration of the Street Fighter II series) featuring the original game and a high definition version using graphics drawn by UDON Entertainment, and arranged music by OverClocked ReMix members.[1] The game was designed by Backbone Entertainment's David Sirlin to be the sixth definitive version of Street Fighter II,[2] although it is in fact the seventh, being released after Hyper Street Fighter II. | END ID: 868

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 869 | TITLE: Diomede Islands | CONTENT: The Big Diomede Island is the easternmost point of Russia. | END ID: 869

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 870 | TITLE: Sonnet | CONTENT: When English sonnets were introduced by Thomas Wyatt (1503–1542) in the early 16th century, his sonnets and those of his contemporary the Earl of Surrey were chiefly translations from the Italian of Petrarch and the French of Ronsard and others. While Wyatt introduced the sonnet into English, it was Surrey who developed the rhyme scheme – abab cdcd efef gg – which now characterizes the English sonnet. Having previously circulated in manuscripts only, both poets' sonnets were first published in Richard Tottel's Songes and Sonnetts, better known as Tottel's Miscellany (1557). | END ID: 870

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 871 | TITLE: Talk:First law of thermodynamics | CONTENT: Using your statement, I took a cut at it. PAR 22:09, 9 November 2005 (UTC) | END ID: 871

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 872 | TITLE: Stolen Valor Act of 2005 | CONTENT: The Act was first introduced in the U.S. House of Representatives on July 19, 2005, by Representative John Salazar, a Democrat from Colorado, as H.R. 3352.[2][3] It was introduced in the Senate by Senator Kent Conrad, a Democrat from North Dakota, on November 10, 2005, as S. 1998.[4][5] The Senate version was passed unanimously on September 7, 2006.[5][6] The House passed the Senate version, S. 1998, on December 6, 2006.[7] | END ID: 872

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 873 | TITLE: Catch-22 | CONTENT: Catch-22 is a satirical novel by American author Joseph Heller. He began writing it in 1953; the novel was first published in 1961. Often cited as one of the most significant novels of the twentieth century,[2] it uses a distinctive non-chronological third-person omniscient narration, describing events from the points of view of different characters. The separate storylines are out of sequence so the timeline develops along with the plot. | END ID: 873

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 874 | TITLE: Brisket | CONTENT: Brisket is a cut of meat from the breast or lower chest of beef or veal. The beef brisket is one of the nine beef primal cuts, though the precise definition of the cut differs internationally. The brisket muscles include the superficial and deep pectorals. As cattle do not have collar bones, these muscles support about 60% of the body weight of standing/moving cattle. This requires a significant amount of connective tissue, so the resulting meat must be cooked correctly to tenderize the connective tissue. | END ID: 874

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 875 | TITLE: Mughal Empire | CONTENT: The beginning of the empire is conventionally dated to the victory by its founder Babur over Ibrahim Lodi, the last ruler of the Delhi Sultanate, in the First Battle of Panipat (1526). The Mughal emperors had roots in the Turco-Mongol Timurid dynasty of Central Asia, claiming direct descent from both Genghis Khan (founder of the Mongol Empire, through his son Chagatai Khan) and Timur (Turco-Mongol conqueror who founded the Timurid Empire). During the reign of Humayun, the successor of Babur, the empire was briefly interrupted by the Sur Empire. The "classic period" of the Mughal Empire started in 1556 with the ascension of Akbar the Great to the throne. Under the rule of Akbar and his son Jahangir, the region enjoyed economic progress as well as religious harmony, and the monarchs were interested in local religious and cultural traditions. Akbar was a successful warrior who also forged alliances with several Hindu Rajput kingdoms. Some Rajput kingdoms continued to pose a significant threat to the Mughal dominance of northwestern India, but most of them were subdued by Akbar. All Mughal emperors were Muslims; Akbar, however, propounded a syncretic religion in the latter part of his life called Dīn-i Ilāhī, as recorded in historical books like Ain-i-Akbari and Dabistān-i Mazāhib.[25] | END ID: 875

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 876 | TITLE: George Washington | CONTENT: After much reluctance, he was persuaded to attend the Constitutional Convention in Philadelphia during the summer of 1787 as a delegate from Virginia, where he was unanimously elected as president of the Convention.[157] He held considerable criticism of the Articles of Confederation of the thirteen colonies, for the weak central government which it established, referring to the Articles as no more than "a rope of sand" to support the new nation.[158] Washington's view for the need of a strong federal government grew out of the recent war, as well as the inability of the Continental Congress to rally the states to provide for the needs of the military, as was clearly demonstrated for him during the winter at Valley Forge. The general populace, however, did not share Washington's views of a strong federal government binding the states together, comparing such a prevailing entity to the British Parliament that previously ruled and taxed the colonies.[159] | END ID: 876

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 877 | TITLE: Fight-or-flight response | CONTENT: In the context of the fight or flight response, emotional regulation is used proactively to avoid threats of stress or to control the level of emotional arousal.[16][17] | END ID: 877

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 878 | TITLE: George Costanza | CONTENT: George often goes to impressive measures to build and maintain his relationships with women. In "The Conversion", he goes through the process of converting to the Latvian Orthodox religion as his girlfriend’s conservative parents would not let her date somebody outside their religion. In "The Susie", he deems it so important that he make a grand entrance at his work’s ball with his attractive girlfriend Allison that, upon finding out that she plans to break up with him, George goes to great lengths to avoid her before the ball, stating "If she can't find me, she can't break up with me." Ultimately though, the one relationship he holds long-term, with his fiancé Susan, is the one about which he is seemingly least enthusiastic, as shown by his ongoing attempts to first postpone, and later cancel, their wedding and his rather nonchalant reaction when she suddenly dies. In fact, in "The Foundation", George shows greater emotion while discussing the death of the Star Trek character Spock in the movie, The Wrath of Khan than after Susan's death. | END ID: 878

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 879 | TITLE: Religion in India | CONTENT: Religion plays a major role in the Indian way of life.[85] Rituals, worship, and other religious activities are very prominent in an individual's daily life; it is also a principal organiser of social life. The degree of religiosity varies amongst individuals; in recent decades, religious orthodoxy and observances have become less common in Indian society, particularly amongst young urban-dwellers. | END ID: 879

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 880 | TITLE: Israeli Declaration of Independence | CONTENT: As leader of the Yishuv, David Ben-Gurion was the first person to sign. The declaration was due to be signed by all 37 members of Moetzet HaAm. However, twelve members could not attend, eleven of them trapped in besieged Jerusalem and one abroad. The remaining 25 signatories present were called up in alphabetical order to sign, leaving spaces for those absent. Although a space was left for him between the signatures of Eliyahu Dobkin and Meir Vilner, Zerach Warhaftig signed at the top of the next column, leading to speculation that Vilner's name had been left alone to isolate him, or to stress that even a communist agreed with the declaration.[15] However, Warhaftig later denied this, stating that a space had been left for him (as he was one of the signatories trapped in Jerusalem) where a Hebraicised form of his name would have fitted alphabetically, but he insisted on signing under his actual name so as to honour his father's memory and so moved down two spaces. He and Vilner would be the last surviving signatories, and remained close for the rest of their lives. Of the signatories, two were women (Golda Meir (Meyerson/Myerson) and Rachel Cohen-Kagan).[18] | END ID: 880

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 881 | TITLE: Dexter (season 3) | CONTENT: While stalking a murderous drug dealer, "Freebo", Dexter stumbles upon a fight between Freebo and another man, whom he is forced to kill in self-defense. This is the first time Dexter kills someone of whose guilt he wasn't completely sure. This victim turns out to be Oscar Prado, brother of Miguel Prado (Jimmy Smits), a prominent assistant district attorney and old flame of Lt. LaGuerta. Miguel comes to confide in and trust Dexter after he helps assist with the case. Dexter tracks down Freebo and kills him, but as he leaves he encounters Miguel, who has followed a lead to Freebo's location. Dexter tells Miguel he had discovered Freebo and killed him in self-defense, but instead of reprimanding him, Miguel thanks him and offers to help him cover up the crime. As Dexter and Miguel cooperate to conceal Freebo's true demise from everyone else, the duo end up becoming close friends, as do Rita and Miguel's wife, Sylvia. | END ID: 881

@@@@@ CHUNK (user, text/plain) @@@@@
ID: 882 | TITLE: Interstellar medium | CONTENT: In the series of investigations, Viktor Ambartsumian introduced the now commonly accepted notion that interstellar matter occurs in the form of clouds.[12] | END ID: 882

@@@@@ CHUNK (user, text/plain) @@@@@
-------------------------------------------------------------------------------
- Here is the end of relavant documents
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
- From now, I will provide some interaction example between user and assistant.
- Between `<` and `>`, it is the explaination by instructor. **You should not repeat them.** They are just instruction for you.
-------------------------------------------------------------------------------

====== Example 1 ======
< User's turn >

What is the answer of the query? Print out the answer of the query. Then format the answers into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: 
where did the dewey decimal system come from

< Assistant's turn >

Draft Answer: ['Unknown']
Verification: ... You have to fill verification your self ...
Final Answer: ['Melvil Dewey']

====== Example 2 ======
< User's turn >

What is the answer of the query? Print out the answer of the query. Then format the answers into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: 
where did the titanic sink at what ocean

< Assistant's turn >

Draft Answer: ['North Atlantic Ocean', 'Pacific Ocean']
Verification: ... You have to fill verification your self ...
Final Answer: ['North Atlantic Ocean']

====== Example 3 ======
< User's turn >

What is the answer of the query? Print out the answer of the query. Then format the answers into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: 
where does the edinburgh fringe festival take place

< Assistant's turn >

Draft Answer: ['in Edinburgh, England']
Verification: ... You have to fill verification your self ...
Final Answer: ['in Edinburgh, Scotland']

====== Example 4 ======
< User's turn >

What is the answer of the query? Print out the answer of the query. Then format the answers into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: 
who headed the 7th central pay commission of india

< Assistant's turn >

Draft Answer: ['Mathur']
Verification: ... You have to fill verification your self ...
Final Answer: ['Justice A.K Mathur']

====== Example 5 ======
< User's turn >

What is the answer of the query? Print out the answer of the query. Then format the answers into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
query: 
what was the code name of the us program to develop an atomic bomb

< Assistant's turn >

Draft Answer: ['New york city']
Verification: ... You have to fill verification your self ...
Final Answer: ['Development of Substitute Materials', 'Manhattan', 'The Manhattan Project']

-------------------------------------------------------------------------------
- From now, let's start talk.
-------------------------------------------------------------------------------

Now you have to answer my query. 

Before answer query, please read following rules.
- Print out the answer of the query. 
- Format the answers into a list.
- If there is no perfect answer output the closest one (you can just guess). 
- Do not give an empty final answer.
- Please be concise when you write the final answer.
- Write anything before "Final Answer:" to help me understand how you answer.
- Write "Draft Answer:" then think about draft answer.
- **Verify your draft answer before write final answer.** If you are just write final answer without verification, then I will fire you.
- Verification should be clear and detailed. You have to give me evidences and reasonings.
- Look provided relavent documents if you want.
- Provide exact date as much as you can.
- The answer must be detailed and clear.

query:
"""

rag_qa_pairs = r"""{"qid": "test1018", "query_text": "when does monday night raw come on hulu", "metadata": {"qrels": [["doc35916", 1]]}, "answers": ["the following day"]}
{"qid": "test103", "query_text": "who did puerto rico belong to before the u.s", "metadata": {"qrels": [["doc3528", 1]]}, "answers": ["Spain", "Taíno", "indigenous Taíno people"]}
{"qid": "test1032", "query_text": "when did season 4 of glee come out", "metadata": {"qrels": [["doc36245", 1]]}, "answers": ["September 13, 2012"]}
{"qid": "test1041", "query_text": "what is the genus of a bald eagle", "metadata": {"qrels": [["doc36467", 1]]}, "answers": ["Haliaeetus"]}
{"qid": "test1062", "query_text": "when does boomer find out she a cylon", "metadata": {"qrels": [["doc36987", 1]]}, "answers": ["Kobol's Last Gleaming"]}
{"qid": "test1065", "query_text": "what is the female lion called in lion king", "metadata": {"qrels": [["doc37055", 1]]}, "answers": ["Nala"]}
{"qid": "test1087", "query_text": "who plays the woodsman in over the garden wall", "metadata": {"qrels": [["doc37964", 1]]}, "answers": ["Christopher Lloyd"]}
{"qid": "test1136", "query_text": "who introduced the first chrismas tree to the uk", "metadata": {"qrels": [["doc40236", 1]]}, "answers": ["Charlotte of Mecklenburg-Strelitz"]}
{"qid": "test1156", "query_text": "who played cruella de vil in 101 dalmatians", "metadata": {"qrels": [["doc41039", 1]]}, "answers": ["Glenn Close"]}
{"qid": "test1172", "query_text": "who sang the song i wanna be sedated", "metadata": {"qrels": [["doc41558", 1]]}, "answers": ["the Ramones", "Joey Romones"]}
{"qid": "test1187", "query_text": "how many players on the line of scrimmage in american football", "metadata": {"qrels": [["doc42023", 1]]}, "answers": ["at least seven", "at least seven players", "seven", "7"]}
{"qid": "test1208", "query_text": "when did the tv show the waltons first air", "metadata": {"qrels": [["doc43104", 1]]}, "answers": ["September 1972", "September 14, 1972"]}
{"qid": "test1223", "query_text": "who is emma dating in once upon a time", "metadata": {"qrels": [["doc43580", 1]]}, "answers": ["Hook"]}
{"qid": "test1283", "query_text": "when was the last time giants won superbowl", "metadata": {"qrels": [["doc45373", 1]]}, "answers": ["2012"]}
{"qid": "test1305", "query_text": "where does the sun hit the us first", "metadata": {"qrels": [["doc46266", 1]]}, "answers": ["the summit of Cadillac Mountain"]}
{"qid": "test1314", "query_text": "who is the authority or governing body of mca", "metadata": {"qrels": [["doc46476", 1]]}, "answers": ["Indian government"]}
{"qid": "test1322", "query_text": "where does prime rib come from on a cow", "metadata": {"qrels": [["doc46823", 1]]}, "answers": ["the primal rib"]}
{"qid": "test1325", "query_text": "who plays joker in batman the dark knight", "metadata": {"qrels": [["doc46886", 1]]}, "answers": ["Ledger"]}
{"qid": "test1331", "query_text": "what is the common name for gravitational force", "metadata": {"qrels": [["doc47145", 1]]}, "answers": ["Gravity", "Gravity, or gravitation"]}
{"qid": "test137", "query_text": "fast and furious 7 red car abu dhabi", "metadata": {"qrels": [["doc5160", 1]]}, "answers": ["The Lykan Hypersport"]}
{"qid": "test1380", "query_text": "who played the original steve mcgarrett on hawaii five-o", "metadata": {"qrels": [["doc48941", 1]]}, "answers": ["Jack Lord", "John Joseph Patrick Ryan"]}
{"qid": "test1434", "query_text": "is aluminium a ferrous or non ferrous metal", "metadata": {"qrels": [["doc51029", 1]]}, "answers": ["non-ferrous"]}
{"qid": "test1439", "query_text": "who has a ring of power in lotr", "metadata": {"qrels": [["doc51161", 1]]}, "answers": ["Sauron"]}
{"qid": "test1459", "query_text": "the era of the great mughals began with the accession of", "metadata": {"qrels": [["doc52008", 1]]}, "answers": ["Akbar the Great", "Babur"]}
{"qid": "test1464", "query_text": "when did hyderabad became a part of india", "metadata": {"qrels": [["doc52200", 1]]}, "answers": ["24 November 1949", "1949"]}
{"qid": "test1575", "query_text": "vikram samvat calender is official in which country", "metadata": {"qrels": [["doc56225", 1]]}, "answers": ["Nepal"]}
{"qid": "test1579", "query_text": "who won the oscar for best picture in 1976", "metadata": {"qrels": [["doc56276", 1]]}, "answers": ["Rocky"]}
{"qid": "test1609", "query_text": "where is the protien made in the cell", "metadata": {"qrels": [["doc56986", 1]]}, "answers": ["cell nucleus", "in the cell nucleus"]}
{"qid": "test1661", "query_text": "what is the tigers name in life of pi", "metadata": {"qrels": [["doc58701", 1]]}, "answers": ["Richard Parker"]}
{"qid": "test1679", "query_text": "who played lionel in as time goes by", "metadata": {"qrels": [["doc59485", 1]]}, "answers": ["Geoffrey Palmer", "Geoffrey Dyson Palmer", "Geoffrey Dyson Palmer, OBE"]}
{"qid": "test172", "query_text": "when does mexico play in the winter olympics", "metadata": {"qrels": [["doc6274", 1]]}, "answers": ["9 to 25 February 2018"]}
{"qid": "test1725", "query_text": "who is the head a in pretty little liars", "metadata": {"qrels": [["doc61143", 1]]}, "answers": ["CeCe Drake"]}
{"qid": "test1843", "query_text": "what does g stand for in ncis los angeles", "metadata": {"qrels": [["doc64877", 1]]}, "answers": ["Grisha"]}
{"qid": "test193", "query_text": "what is the membrane on the surface of the stomach called", "metadata": {"qrels": [["doc7074", 1]]}, "answers": ["Serous Membrane", "the visceral membrane"]}
{"qid": "test1959", "query_text": "when did it become law to stand for the national anthem", "metadata": {"qrels": [["doc68843", 1]]}, "answers": ["June 22, 1942"]}
{"qid": "test1963", "query_text": "what is the name of the compound p4010", "metadata": {"qrels": [["doc69002", 1]]}, "answers": ["Phosphorus pentoxide"]}
{"qid": "test1989", "query_text": "krypton-85 decays by emission of a beta particle. the product of this decay is", "metadata": {"qrels": [["doc69765", 1]]}, "answers": ["rubidium-85", "Rb-85"]}
{"qid": "test2013", "query_text": "which episode does gideon die in criminal minds", "metadata": {"qrels": [["doc70508", 1]]}, "answers": ["\"Nelson's Sparrow\"", "Nelson's Sparrow"]}
{"qid": "test2213", "query_text": "first jnanpith award was an autor of which language", "metadata": {"qrels": [["doc76971", 1]]}, "answers": ["Malayalam"]}
{"qid": "test2367", "query_text": "who was the first signatory of the israeli declaration of independence", "metadata": {"qrels": [["doc81884", 1]]}, "answers": ["David Ben-Gurion"]}
{"qid": "test2407", "query_text": "where did the rulers of the qing dynasty originate", "metadata": {"qrels": [["doc83148", 1]]}, "answers": ["Manchuria"]}
{"qid": "test2425", "query_text": "when did where are you now come out", "metadata": {"qrels": [["doc83761", 1]]}, "answers": ["February 27, 2015", "February 27, 2015"]}
{"qid": "test2492", "query_text": "who made the song we are the world", "metadata": {"qrels": [["doc86045", 1]]}, "answers": ["produced by Quincy Jones", "USA for Africa"]}
{"qid": "test2506", "query_text": "who sings the rap in baby by justin bieber", "metadata": {"qrels": [["doc86497", 1]]}, "answers": ["Ludacris"]}
{"qid": "test2508", "query_text": "jawaharlal nehru centre for advanced scientific research jakkur campus", "metadata": {"qrels": [["doc86529", 1]]}, "answers": ["Jakkur", "Jakkur, Bangalore, India"]}
{"qid": "test2649", "query_text": "why does cooling water run through the condenser", "metadata": {"qrels": [["doc90702", 1]]}, "answers": ["condense the steam"]}
{"qid": "test2650", "query_text": "when does a cell have condensed visible chromosomes also known as sister chromatids", "metadata": {"qrels": [["doc90765", 1]]}, "answers": ["metaphase"]}
{"qid": "test2655", "query_text": "what was hawaii's primary export to the united states", "metadata": {"qrels": [["doc91034", 1]]}, "answers": ["coffee", "honey", "livestock", "macadamia nuts", "pineapple", "sugarcane"]}
{"qid": "test269", "query_text": "what type of government did the ming dynasty have", "metadata": {"qrels": [["doc10019", 1]]}, "answers": ["imperial rule", "imperial"]}
{"qid": "test2745", "query_text": "what is the filename extension used for all java source files", "metadata": {"qrels": [["doc94476", 1]]}, "answers": [".java"]}
{"qid": "test2747", "query_text": "when was the last time unc did not make the ncaa tournament", "metadata": {"qrels": [["doc94595", 1]]}, "answers": ["2003"]}
{"qid": "test2759", "query_text": "different ways to spell corey for a boy", "metadata": {"qrels": [["doc94996", 1]]}, "answers": ["Coire", "Corey", "Corie", "Correy", "Corrie", "Cory", "Khouri", "Kori", "Kory"]}
{"qid": "test2786", "query_text": "what percentage of the population is naturally blonde", "metadata": {"qrels": [["doc95660", 1]]}, "answers": ["2%"]}
{"qid": "test2809", "query_text": "who plays poppy in the beat goes on", "metadata": {"qrels": [["doc96520", 1]]}, "answers": ["Amanda Leighton"]}
{"qid": "test2852", "query_text": "a town in west yorkshire on the river aire home to a rugby league team", "metadata": {"qrels": [["doc97666", 1]]}, "answers": ["Castleford"]}
{"qid": "test2870", "query_text": "what is alpha centauri's approximate distance from earth", "metadata": {"qrels": [["doc98149", 1]]}, "answers": ["4.37 light-years"]}
{"qid": "test2964", "query_text": "how many seasons of the rugrats are there", "metadata": {"qrels": [["doc101357", 1]]}, "answers": ["9", "9 seasons"]}
{"qid": "test297", "query_text": "what is the democracy of the united states", "metadata": {"qrels": [["doc11181", 1]]}, "answers": ["federal republic"]}
{"qid": "test2970", "query_text": "when did cristiano ronaldo go to manchester united", "metadata": {"qrels": [["doc101768", 1]]}, "answers": ["2003", "at age 18 in 2003"]}
{"qid": "test2971", "query_text": "when did the nfl adopt a salary cap", "metadata": {"qrels": [["doc101868", 1]]}, "answers": ["1994", "1994 season", "for the 1994 season", "the 1994 season"]}
{"qid": "test2991", "query_text": "where is the tibia and fibula bone located", "metadata": {"qrels": [["doc102543", 1]]}, "answers": ["leg"]}
{"qid": "test3023", "query_text": "where is the villa in call me by your name", "metadata": {"qrels": [["doc103666", 1]]}, "answers": ["Moscazzano"]}
{"qid": "test3027", "query_text": "when does agents of shield season five start", "metadata": {"qrels": [["doc103787", 1]]}, "answers": ["December 1, 2017"]}
{"qid": "test3066", "query_text": "who said i'll gladly pay you tuesday", "metadata": {"qrels": [["doc104923", 1]]}, "answers": ["Wimpy"]}
{"qid": "test3095", "query_text": "who wrote catch 22 (both names)", "metadata": {"qrels": [["doc105954", 1]]}, "answers": ["American author Joseph Heller", "Joseph Heller", "Joseph Heller."]}
{"qid": "test3124", "query_text": "where does the synthesis of new dna from existing dna occurs", "metadata": {"qrels": [["doc106826", 1]]}, "answers": ["origins of replication", "nucleus"]}
{"qid": "test3129", "query_text": "who does tyler end up with in you get me", "metadata": {"qrels": [["doc106940", 1]]}, "answers": ["Ali"]}
{"qid": "test3139", "query_text": "who was the editor of the journal jugantor published in the time of swadeshi movement", "metadata": {"qrels": [["doc107145", 1]]}, "answers": ["Bhupendranath Dutt"]}
{"qid": "test3155", "query_text": "how and why were serial novels a phenomenon in the nineteenth century", "metadata": {"qrels": [["doc107756", 1]]}, "answers": ["improved economics of distribution", "technological advances in printing", "the rise of literacy"]}
{"qid": "test3198", "query_text": "when did the sat become out of 1600", "metadata": {"qrels": [["doc109307", 1]]}, "answers": ["2014", "2016", "March 2016"]}
{"qid": "test3249", "query_text": "what type of energy do satellites generally use to communicate with earth", "metadata": {"qrels": [["doc110977", 1]]}, "answers": ["electromagnetic waves", "radio and microwave frequencies", "radio frequency"]}
{"qid": "test3322", "query_text": "which layer of the osi model handles physical addressing", "metadata": {"qrels": [["doc113413", 1]]}, "answers": ["physical layer or layer 1", "data link layer", "layer 2"]}
{"qid": "test3332", "query_text": "roman god of the underworld also called orcus or pluto", "metadata": {"qrels": [["doc113775", 1]]}, "answers": ["Dis Pater"]}
{"qid": "test3333", "query_text": "who is regarded as the founder of psychoanalysis", "metadata": {"qrels": [["doc113858", 1]]}, "answers": ["Austrian neurologist Sigmund Freud", "Sigmund Freud"]}
{"qid": "test352", "query_text": "how many national parks are present in india", "metadata": {"qrels": [["doc13434", 1]]}, "answers": ["103"]}
{"qid": "test359", "query_text": "when did they start 3 pointers in basketball", "metadata": {"qrels": [["doc13785", 1]]}, "answers": ["1945", "1961", "1967", "1979"]}
{"qid": "test376", "query_text": "what type of legal system is used in the uk", "metadata": {"qrels": [["doc14159", 1]]}, "answers": ["English law", "Northern Ireland law", "Scots law"]}
{"qid": "test435", "query_text": "how many customers does edf have in the uk", "metadata": {"qrels": [["doc16130", 1]]}, "answers": ["5.7 million", "5.7 million customer accounts"]}
{"qid": "test436", "query_text": "what is the highest peak in the ozarks", "metadata": {"qrels": [["doc16157", 1]]}, "answers": ["Buffalo Lookout", "Lookout"]}
{"qid": "test445", "query_text": "who is the owner of the crowne plaza", "metadata": {"qrels": [["doc16542", 1]]}, "answers": ["InterContinental Hotels Group"]}
{"qid": "test453", "query_text": "batman and robin episode only fools and horses", "metadata": {"qrels": [["doc16745", 1]]}, "answers": ["\"Heroes and Villains\""]}
{"qid": "test492", "query_text": "when was the immigration act passed in canada", "metadata": {"qrels": [["doc18225", 1]]}, "answers": ["1923"]}
{"qid": "test596", "query_text": "who is known as the father of indian constitution", "metadata": {"qrels": [["doc21271", 1]]}, "answers": ["Bhimrao Ramji Ambedkar", "B. R. Ambedkar", "B.R. Ambedkar"]}
{"qid": "test602", "query_text": "who were the judges on dancing on ice 2014", "metadata": {"qrels": [["doc21533", 1]]}, "answers": ["Ashley Roberts", "Jason Gardiner", "Karen Barber", "Robin Cousins"]}
{"qid": "test619", "query_text": "where does a brisket come from on a cow", "metadata": {"qrels": [["doc22116", 1]]}, "answers": ["breast or lower chest", "the breast or lower chest"]}
{"qid": "test623", "query_text": "how long were the pyramids the tallest structure", "metadata": {"qrels": [["doc22219", 1]]}, "answers": ["over 3,800", "over 3,800 years"]}
{"qid": "test641", "query_text": "who played little ricky on i love lucy show", "metadata": {"qrels": [["doc22939", 1]]}, "answers": ["Keith Thibodeaux"]}
{"qid": "test648", "query_text": "when does the new season on the 100 come out", "metadata": {"qrels": [["doc23040", 1]]}, "answers": ["April 24, 2018", "2018"]}
{"qid": "test664", "query_text": "who has won the eurovision song contest the most times", "metadata": {"qrels": [["doc23597", 1]]}, "answers": ["Ireland", "Ireland's Johnny Logan"]}
{"qid": "test668", "query_text": "where do they film young and the restless", "metadata": {"qrels": [["doc23926", 1]]}, "answers": ["CBS Television City", "Los Angeles"]}
{"qid": "test739", "query_text": "list of books written by abul kalam azad", "metadata": {"qrels": [["doc26407", 1]]}, "answers": ["Ghubar-e-Khatir", "India Wins Freedom", "Tarjumanul Quran", "Tazkirah"]}
{"qid": "test741", "query_text": "when does rick and morty play on tv", "metadata": {"qrels": [["doc26484", 1]]}, "answers": ["late-night"]}
{"qid": "test835", "query_text": "who played jennifer in back to the future", "metadata": {"qrels": [["doc30158", 1]]}, "answers": ["Claudia Grace Wells", "Claudia Wells"]}
{"qid": "test884", "query_text": "what type of writing did ancient egypt use", "metadata": {"qrels": [["doc31575", 1]]}, "answers": ["Egyptian hieroglyphs", "hieroglyphs"]}
{"qid": "test90", "query_text": "where does the donkey talk in the bible", "metadata": {"qrels": [["doc2961", 1]]}, "answers": ["Numbers 22:28", "Numbers 22:22-35", "Moab"]}
{"qid": "test902", "query_text": "when did the bill of rights come out", "metadata": {"qrels": [["doc32115", 1]]}, "answers": ["16 December 1689", "1689"]}
{"qid": "test948", "query_text": "who was selected for the 2018 football hall of fame", "metadata": {"qrels": [["doc33654", 1]]}, "answers": ["Bobby Beathard", "Brian Dawkins", "Brian Urlacher", "Jerry Kramer", "Randy Moss", "Ray Lewis", "Robert Brazile", "Terrell Owens"]}
{"qid": "test97", "query_text": "when was you'll never walk alone first released", "metadata": {"qrels": [["doc3100", 1]]}, "answers": ["1945"]}
{"qid": "test971", "query_text": "where was the remake of dirty dancing filmed", "metadata": {"qrels": [["doc34207", 1]]}, "answers": ["Hendersonville, North Carolina", "Hendersonville", "High Hampton Inn in Cashiers", "western North Carolina"]}
{"qid": "test999", "query_text": "where did huntington's disease get its name", "metadata": {"qrels": [["doc35283", 1]]}, "answers": ["the physician George Huntington", "George Huntington"]}"""

prefix = "\n".join(filter(lambda x: not x.startswith('@@@@'), prefix.split('\n')))
rag_qa_pairs = list(map(lambda line: json.loads(line), rag_qa_pairs.split('\n')))