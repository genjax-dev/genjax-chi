Search.setIndex({"docnames": ["genjax/c_interface", "genjax/combinators", "genjax/diff_jl", "genjax/gen_fn", "genjax/interface", "genjax/tour", "genjax/wasm_interface", "index"], "filenames": ["genjax/c_interface.rst", "genjax/combinators.rst", "genjax/diff_jl.rst", "genjax/gen_fn.rst", "genjax/interface.rst", "genjax/tour.rst", "genjax/wasm_interface.rst", "index.rst"], "titles": ["Exposing C++ generative functions", "Generative function combinators", "Diffing with Gen.jl", "What is a generative function?", "Generative function interface", "A tour of fundamentals", "From GenJAX to WebAssembly", "Index"], "terms": {"while": [0, 2], "genjax": [0, 1, 2, 4, 5, 7], "i": [0, 1, 2, 4, 5, 6, 7], "fast": 0, "gpu": 0, "courtesi": 0, "jax": [0, 1, 2, 3, 4, 5, 6, 7], "sometim": 0, "we": [0, 2, 3, 5], "want": [0, 7], "lower": [0, 3], "level": 0, "control": [0, 1], "over": [0, 1, 5, 7], "implement": [0, 1, 2, 3, 4], "our": [0, 3], "emit": 0, "code": [0, 2, 3], "which": [0, 1, 2, 3, 4, 5, 7], "optim": 0, "cpu": 0, "devic": [0, 2], "so": [0, 5], "wish": 0, "leverag": 0, "modern": 0, "compil": [0, 7], "toolchain": 0, "like": [0, 1, 2, 7], "llvm": 0, "fortun": 0, "can": [0, 1, 2, 3, 5], "written": [0, 2], "util": [0, 3], "an": [0, 1, 3, 4], "xla": [0, 2], "primit": [0, 2, 3], "call": [0, 1, 3, 4, 5], "custom_cal": 0, "In": [0, 1, 2, 3], "thi": [0, 1, 2, 3, 4, 5, 6], "note": 0, "ll": [0, 7], "walk": 0, "through": 0, "process": [0, 2], "us": [0, 1, 2, 3, 5], "gentl": 0, "base": [0, 1], "excel": 0, "dan": 0, "foreman": 0, "mackei": 0, "To": [0, 7], "need": [0, 2, 5], "setup": 0, "pybind11": 0, "built": [0, 6], "cmake": 0, "ve": [0, 3], "done": 0, "directori": 0, "ani": [0, 2, 5], "support": [0, 1, 2, 3, 4, 6], "infrastructur": 0, "includ": [0, 1, 2, 3], "header": 0, "onli": [0, 1], "librari": 0, "provid": [0, 1, 3], "tool": 0, "defin": [0, 3, 4], "gen_fn": [0, 1, 3], "h": [0, 3], "": [0, 1, 2, 3, 5, 7], "sketch": 0, "lib": 0, "cpp": 0, "templat": 0, "method": [0, 1, 3, 4], "modul": [1, 4, 6, 7], "hold": 1, "set": [1, 2, 3, 4], "These": [1, 7], "accept": [1, 2, 3, 4], "argument": [1, 2, 3, 4, 5], "return": [1, 2, 3, 4, 5], "modifi": 1, "choic": [1, 2, 4, 5], "shape": [1, 2, 3], "behavior": 1, "thei": [1, 4, 5], "ar": [1, 2, 3, 4, 5, 7], "express": [1, 2, 3, 7], "common": 1, "pattern": [1, 3], "comput": [1, 2, 3, 4, 7], "els": 1, "switchcombin": 1, "across": 1, "vectori": 1, "mapcombin": [1, 5], "depend": 1, "loop": [1, 2], "unfoldcombin": 1, "A": [1, 3, 4, 7], "sharp": 1, "bit": [1, 3], "restrict": 1, "The": [1, 4, 5], "similar": [1, 5], "those": [1, 2], "gen": [1, 3, 4, 5, 7], "jl": [1, 3, 7], "impos": 1, "extra": 1, "construct": [1, 2, 7], "usag": [1, 6], "contrast": 1, "must": 1, "have": 1, "number": [1, 5], "step": 1, "specifi": 1, "ahead": [1, 2], "time": [1, 2], "static": [1, 2, 3], "constant": [1, 3, 5], "length": [1, 2], "chain": [1, 2], "cannot": 1, "variabl": [1, 3, 5], "whose": 1, "valu": [1, 4], "known": [1, 5], "runtim": [1, 2], "similarli": 1, "trace": [1, 3, 4, 5], "due": 1, "fundament": [1, 7], "program": [1, 2, 3, 5, 6, 7], "model": [1, 2, 3, 4, 5], "stand": 1, "current": [1, 2], "allow": [1, 2, 6], "unrol": 1, "flow": 1, "act": 1, "kernel": 1, "previou": 1, "output": 1, "input": 1, "class": [1, 4], "generativefunct": [1, 4], "max_length": 1, "int": 1, "sourc": [1, 3, 4], "singl": 1, "how": [1, 3], "mani": [1, 5], "iter": 1, "run": 1, "one": [1, 5, 6], "same": 1, "signatur": 1, "under": [1, 3, 4], "hood": [1, 3], "lax": 1, "scan": 1, "ha": [1, 2, 3], "requir": [1, 2], "paramet": [1, 4], "instanc": [1, 2, 4, 6], "integ": 1, "applic": 1, "interfac": [1, 2, 3, 7], "perform": [1, 2, 5], "pass": [1, 5], "next": 1, "programm": [1, 2, 3, 4], "initi": 1, "start": [1, 5, 7], "off": [1, 2], "type": 1, "exampl": [1, 4, 5], "import": [1, 3, 4, 5], "def": [1, 3, 4, 5], "random_walk": 1, "kei": [1, 3, 4, 5], "prev": 1, "x": [1, 3, 4], "normal": [1, 2, 3, 4, 5], "1": [1, 3, 4, 5], "0": [1, 3, 4, 5], "1000": 1, "init": 1, "5": [1, 5], "random": [1, 3, 4, 5], "prngkei": [1, 3, 4], "314159": [1, 3, 4], "tr": [1, 4], "jit": [1, 6, 7], "simul": [1, 3, 4], "print": [1, 3, 4], "unfoldtrac": 1, "builtingenerativefunct": [1, 3, 4, 5], "vectorchoicemap": 1, "subtrac": 1, "builtin": [1, 4], "datatyp": [1, 4], "builtintrac": [1, 4], "distribut": [1, 2, 4], "distributiontrac": [1, 4], "_normal": [1, 3, 4], "f32": [1, 3, 4], "score": [1, 4, 5], "9247955": 1, "8264294": 1, "8033758": 1, "2785668": 1, "2": [1, 3], "0266647": 1, "6825668": 1, "2227434": 1, "0590267": 1, "3": 1, "1286416": 1, "97036177": 1, "1971071": 1, "9330346": 1, "9295393": 1, "9236511": 1, "1370858": 1, "0736619": 1, "3489403": 1, "216788": 1, "2314839": 1, "964772": 1, "505276": 1, "0950221": 1, "9406347": 1, "94001603": 1, "8975222": 1, "9548435": 1, "9776876": 1, "98727185": 1, "0896006": 1, "92114383": 1, "9103003": 1, "3686922": 1, "0431892": 1, "9449425": 1, "93848586": 1, "9802654": 1, "7054317": 1, "1110342": 1, "2191311": 1, "9594871": 1, "921643": 1, "1974747": 1, "9208673": 1, "8259932": 1, "3640103": 1, "5634525": 1, "0195951": 1, "29763": 1, "1016977": 1, "0665258": 1, "9322612": 1, "305265": 1, "0323236": 1, "2881626": 1, "9774628": 1, "9370697": 1, "99197257": 1, "740237": 1, "0095206": 1, "0749516": 1, "9412691": 1, "5821121": 1, "9249668": 1, "9306303": 1, "97086483": 1, "97936726": 1, "047364": 1, "9200479": 1, "0187916": 1, "94722944": 1, "99590576": 1, "9190724": 1, "5056224": 1, "5752995": 1, "0594754": 1, "97187924": 1, "0011059": 1, "3966688": 1, "5454192": 1, "91909355": 1, "9236219": 1, "3604195": 1, "3297931": 1, "5446706": 1, "4262666": 1, "0913982": 1, "9689624": 1, "7285683": 1, "0678089": 1, "2547052": 1, "9275869": 1, "332803": 1, "95874155": 1, "7133632": 1, "0067157": 1, "1248052": 1, "92415434": 1, "0039259": 1, "0435245": 1, "9434873": 1, "2679868": 1, "9473176": 1, "91893923": 1, "9287169": 1, "1307158": 1, "1587722": 1, "9337518": 1, "1335533": 1, "1165951": 1, "2971752": 1, "1801987": 1, "0292981": 1, "2367817": 1, "8744936": 1, "2753654": 1, "1985173": 1, "91908145": 1, "9292165": 1, "95536935": 1, "2310364": 1, "9701659": 1, "6035593": 1, "2774913": 1, "94313437": 1, "115056": 1, "5195138": 1, "1955721": 1, "92698014": 1, "0104014": 1, "9889713": 1, "3049183": 1, "0862991": 1, "9215293": 1, "0079668": 1, "165556": 1, "4652481": 1, "0654192": 1, "98004955": 1, "2117503": 1, "1789124": 1, "223921": 1, "9217562": 1, "92392826": 1, "9587747": 1, "9404258": 1, "0554875": 1, "208194": 1, "1479881": 1, "9318459": 1, "5629766": 1, "9924774": 1, "0041417": 1, "5514216": 1, "9535184": 1, "9191688": 1, "948944": 1, "1659278": 1, "4763277": 1, "96144444": 1, "0097182": 1, "4340658": 1, "0567257": 1, "106851": 1, "9259598": 1, "34997": 1, "0128845": 1, "9268742": 1, "5610677": 1, "5439048": 1, "93395525": 1, "9393441": 1, "0303279": 1, "7087497": 1, "9229465": 1, "9359642": 1, "3775138": 1, "3778993": 1, "5344801": 1, "1434014": 1, "9451097": 1, "6202878": 1, "5075905": 1, "92839426": 1, "96546394": 1, "1400199": 1, "9780989": 1, "2261438": 1, "94282126": 1, "0239792": 1, "92691": 1, "5112576": 1, "5411043": 1, "2123536": 1, "4036967": 1, "0634372": 1, "1718065": 1, "3191869": 1, "1150541": 1, "4172748": 1, "91957176": 1, "2709292": 1, "924644": 1, "2852697": 1, "272821": 1, "7166464": 1, "96666664": 1, "0890166": 1, "0468526": 1, "3350427": 1, "91903204": 1, "2685387": 1, "92042524": 1, "8025808": 1, "9489279": 1, "2425568": 1, "9924948": 1, "7841061": 1, "500284": 1, "331636": 1, "0278989": 1, "4505509": 1, "2511759": 1, "8678237": 1, "94875014": 1, "9509466": 1, "3166108": 1, "9857655": 1, "94353354": 1, "1696179": 1, "0360458": 1, "0413551": 1, "0088977": 1, "029573": 1, "3479209": 1, "073021": 1, "9394914": 1, "1473044": 1, "0907407": 1, "2431202": 1, "9874072": 1, "3357928": 1, "91957456": 1, "9217661": 1, "3781476": 1, "0084717": 1, "816621": 1, "9279082": 1, "95476973": 1, "6009717": 1, "6484233": 1, "336008": 1, "142037": 1, "9431556": 1, "5090656": 1, "0059848": 1, "2493654": 1, "2850779": 1, "1462996": 1, "0942909": 1, "2331893": 1, "9477549": 1, "1114495": 1, "7686689": 1, "7536306": 1, "0777172": 1, "0386844": 1, "7763121": 1, "9264138": 1, "1817181": 1, "8518555": 1, "5322287": 1, "5148873": 1, "9772895": 1, "0299677": 1, "2961234": 1, "93199414": 1, "1063368": 1, "9915861": 1, "9845294": 1, "977413": 1, "0052937": 1, "1403036": 1, "3495955": 1, "9338312": 1, "9285566": 1, "003109": 1, "2023823": 1, "0852224": 1, "124919": 1, "92195904": 1, "2511541": 1, "821518": 1, "95788914": 1, "9491408": 1, "543939": 1, "9204632": 1, "9513795": 1, "4437249": 1, "6898186": 1, "1438174": 1, "2752304": 1, "8298466": 1, "4084107": 1, "0262048": 1, "1197019": 1, "3335649": 1, "9587564": 1, "9225472": 1, "9393435": 1, "3882489": 1, "3671358": 1, "9207077": 1, "9948641": 1, "0075154": 1, "331109": 1, "919828": 1, "0669774": 1, "99951464": 1, "1709476": 1, "4668742": 1, "5968702": 1, "192229": 1, "3831625": 1, "1914823": 1, "6090014": 1, "0386168": 1, "8706216": 1, "1897192": 1, "2177794": 1, "908894": 1, "4928322": 1, "98092586": 1, "5152133": 1, "9463507": 1, "92175883": 1, "0329885": 1, "453642": 1, "2970442": 1, "9191012": 1, "048263": 1, "1734096": 1, "4501321": 1, "0246925": 1, "4917169": 1, "9193713": 1, "9337553": 1, "4271564": 1, "5532604": 1, "3233856": 1, "048869": 1, "8414493": 1, "6880481": 1, "4676492": 1, "472128": 1, "7624578": 1, "1144087": 1, "9436233": 1, "2421441": 1, "0836378": 1, "1609774": 1, "0881034": 1, "0224075": 1, "92424095": 1, "1064063": 1, "9240038": 1, "2548755": 1, "1185275": 1, "2715448": 1, "919816": 1, "9933098": 1, "5333135": 1, "0681254": 1, "0160403": 1, "0601673": 1, "9270407": 1, "9318489": 1, "91955173": 1, "2928594": 1, "6412485": 1, "9269065": 1, "1550006": 1, "4966414": 1, "92574495": 1, "0855591": 1, "4152812": 1, "9827776": 1, "7836821": 1, "0182176": 1, "9209466": 1, "0034442": 1, "9981484": 1, "339892": 1, "2611157": 1, "4844784": 1, "5192733": 1, "434887": 1, "92465043": 1, "1437145": 1, "4896731": 1, "5882196": 1, "09127": 1, "9307126": 1, "6377085": 1, "3000256": 1, "976712": 1, "6039705": 1, "0432205": 1, "0974997": 1, "9269077": 1, "5150778": 1, "210549": 1, "0622867": 1, "0148045": 1, "9777867": 1, "0049224": 1, "7541101": 1, "98954386": 1, "0639665": 1, "95254546": 1, "744336": 1, "0471963": 1, "92023903": 1, "5826888": 1, "9440527": 1, "9705187": 1, "7442542": 1, "0683318": 1, "0608697": 1, "9189401": 1, "9651224": 1, "9586676": 1, "2073822": 1, "120968": 1, "3207738": 1, "4714401": 1, "9048508": 1, "1594034": 1, "5413487": 1, "1426531": 1, "3932105": 1, "9940948": 1, "0830255": 1, "095564": 1, "2601744": 1, "9323352": 1, "2410731": 1, "94299006": 1, "6753691": 1, "726385": 1, "7349787": 1, "0807834": 1, "95001423": 1, "92910874": 1, "179873": 1, "1630319": 1, "7655227": 1, "765959": 1, "9406285": 1, "2274059": 1, "5794697": 1, "9917327": 1, "9223109": 1, "0914738": 1, "4551196": 1, "2657688": 1, "2383995": 1, "9776958": 1, "1555552": 1, "0518377": 1, "9312515": 1, "9197242": 1, "7818896": 1, "1618724": 1, "9198347": 1, "4684944": 1, "9965103": 1, "3408415": 1, "8628223": 1, "161858": 1, "9312977": 1, "6368673": 1, "97507435": 1, "2418422": 1, "060536": 1, "06771": 1, "0115889": 1, "7355707": 1, "9964187": 1, "179048": 1, "013026": 1, "9702502": 1, "9328655": 1, "0770333": 1, "1494777": 1, "9191759": 1, "2078505": 1, "1641474": 1, "9388316": 1, "2113551": 1, "1720588": 1, "1006536": 1, "9203224": 1, "6395209": 1, "94835377": 1, "0236363": 1, "9803456": 1, "94751734": 1, "9797116": 1, "9696385": 1, "6115072": 1, "208239": 1, "9190829": 1, "977862": 1, "98193663": 1, "9268882": 1, "0204929": 1, "91994417": 1, "9531969": 1, "9189511": 1, "9398222": 1, "0947157": 1, "6356655": 1, "0463225": 1, "9264907": 1, "6758392": 1, "0330423": 1, "9681345": 1, "6363451": 1, "7563177": 1, "2078285": 1, "9238097": 1, "3960532": 1, "1069452": 1, "3920498": 1, "3640983": 1, "9270871": 1, "4688567": 1, "1352429": 1, "2052292": 1, "079512": 1, "0824822": 1, "1009946": 1, "3434997": 1, "5463296": 1, "92041856": 1, "5713286": 1, "0127304": 1, "0920502": 1, "3073483": 1, "91963863": 1, "1521417": 1, "1212919": 1, "7929527": 1, "9237704": 1, "92072266": 1, "9247192": 1, "0072103": 1, "2839991": 1, "0163696": 1, "0911306": 1, "96852684": 1, "1734395": 1, "3538206": 1, "6411132": 1, "011904": 1, "8689435": 1, "0400534": 1, "0130743": 1, "0677464": 1, "8959175": 1, "0150979": 1, "98035634": 1, "0000938": 1, "2459164": 1, "1006167": 1, "9427314": 1, "96994454": 1, "9309763": 1, "3874879": 1, "211853": 1, "5940104": 1, "9234709": 1, "91945773": 1, "2585943": 1, "9348399": 1, "2842795": 1, "3124999": 1, "4309962": 1, "7587657": 1, "0271717": 1, "98227966": 1, "9190737": 1, "6992042": 1, "0708077": 1, "1634164": 1, "9769164": 1, "94542086": 1, "6884432": 1, "3853079": 1, "9620889": 1, "1466463": 1, "7219144": 1, "021246": 1, "1076897": 1, "0057255": 1, "1840448": 1, "3184133": 1, "3476735": 1, "1842115": 1, "5635514": 1, "94229734": 1, "9035817": 1, "9964495": 1, "1343832": 1, "6735498": 1, "9704248": 1, "0181334": 1, "92647946": 1, "91928524": 1, "9565817": 1, "9519905": 1, "2933114": 1, "2784772": 1, "93793535": 1, "9844099": 1, "92324036": 1, "6783798": 1, "9763152": 1, "5922506": 1, "0974678": 1, "4767641": 1, "994304": 1, "3029319": 1, "93305296": 1, "5260262": 1, "1857448": 1, "9228429": 1, "927602": 1, "3880548": 1, "1967702": 1, "2983024": 1, "1675909": 1, "1843731": 1, "1005491": 1, "9072065": 1, "94130546": 1, "94355136": 1, "6309623": 1, "339983": 1, "9332317": 1, "0928097": 1, "4543555": 1, "8555745": 1, "0742635": 1, "0748924": 1, "9280076": 1, "5493307": 1, "9478842": 1, "5275909": 1, "91905564": 1, "8657334": 1, "09952": 1, "92757964": 1, "93194455": 1, "96436185": 1, "0787486": 1, "106803": 1, "93968827": 1, "0433532": 1, "4": 1, "8005548": 1, "96588963": 1, "3915883": 1, "0401858": 1, "4239559": 1, "4542983": 1, "4564418": 1, "8411291": 1, "9417193": 1, "1910346": 1, "5830448": 1, "9350597": 1, "163468": 1, "9318762": 1, "96212363": 1, "92639095": 1, "7487792": 1, "9318532": 1, "1255169": 1, "1880305": 1, "9742923": 1, "5345426": 1, "0295765": 1, "3494875": 1, "4862223": 1, "1539454": 1, "0942886": 1, "91900396": 1, "0719556": 1, "95050496": 1, "263759": 1, "20016": 1, "727239": 1, "4119223": 1, "2663679": 1, "9483399": 1, "0113554": 1, "0867366": 1, "3314208": 1, "9826516": 1, "0493113": 1, "0434297": 1, "0288161": 1, "508704": 1, "0745921": 1, "4681069": 1, "9270997": 1, "6175504": 1, "664446": 1, "5099747": 1, "4300222": 1, "3544656": 1, "1932354": 1, "92646426": 1, "3330492": 1, "3553838": 1, "9354234": 1, "4900315": 1, "9448937": 1, "7299422": 1, "0302775": 1, "4873276": 1, "0715886": 1, "4839308": 1, "6147878": 1, "9601309": 1, "0750105": 1, "9738642": 1, "97490096": 1, "3886573": 1, "0434636": 1, "9705856": 1, "4139177": 1, "2771312": 1, "7094725": 1, "2460672": 1, "0701592": 1, "3908534": 1, "0630542": 1, "956801": 1, "1187638": 1, "4395136": 1, "5672214": 1, "7320764": 1, "4632607": 1, "213386": 1, "973386": 1, "488953": 1, "2796662": 1, "1096613": 1, "003006": 1, "9546442": 1, "9195074": 1, "0323304": 1, "9697499": 1, "0635805": 1, "014695": 1, "9602997": 1, "3334137": 1, "9714752": 1, "6826369": 1, "7793794": 1, "3662695": 1, "9980285": 1, "9196106": 1, "1517055": 1, "1779733": 1, "0526433": 1, "1128006": 1, "0864367": 1, "2475104": 1, "9592639": 1, "9932587": 1, "1065582": 1, "435891": 1, "8119606": 1, "1143833": 1, "1009586": 1, "6472051": 1, "2042428": 1, "97634494": 1, "928174": 1, "92753536": 1, "93592554": 1, "0540941": 1, "9203523": 1, "0797648": 1, "0139843": 1, "0679173": 1, "1585246": 1, "567631": 1, "9204588": 1, "9321702": 1, "3992858": 1, "158102": 1, "9688189": 1, "971456": 1, "878445": 1, "972546": 1, "9727127": 1, "0175854": 1, "92090535": 1, "0920682": 1, "4643493": 1, "0762011": 1, "4119687": 1, "9248072": 1, "543763": 1, "4893298": 1, "0384464": 1, "7985811": 1, "5625026": 1, "9729093": 1, "046299": 1, "2895416": 1, "8969216": 1, "92146015": 1, "022398": 1, "0129713": 1, "765014": 1, "4011815": 1, "1333921": 1, "7683569": 1, "3641732": 1, "9556438": 1, "9283344": 1, "0051389": 1, "9850212": 1, "93290085": 1, "710005": 1, "9361688": 1, "9193044": 1, "1958956": 1, "8846436": 1, "3092307": 1, "2962965": 1, "3106947": 1, "95493585": 1, "4222167": 1, "5909841": 1, "9372151": 1, "0083878": 1, "931547": 1, "54373": 1, "7465079": 1, "2805067": 1, "15201": 1, "01106": 1, "4169915": 1, "9453222": 1, "3836294": 1, "93706954": 1, "1689323": 1, "94790673": 1, "2255241": 1, "003263": 1, "5499873": 1, "9214103": 1, "0730889": 1, "190378": 1, "0907799": 1, "1637406": 1, "5364721": 1, "0702175": 1, "0811101": 1, "1131618": 1, "2124925": 1, "3037463": 1, "1747557": 1, "07532": 1, "1401087": 1, "6184784": 1, "1460532": 1, "2622583": 1, "9813436": 1, "93435353": 1, "2620931": 1, "3567553": 1, "490694": 1, "7895718": 1, "92520916": 1, "3656213": 1, "933106": 1, "93421054": 1, "9316754": 1, "1408489": 1, "9855544": 1, "0286016": 1, "8049772": 1, "1461599": 1, "3672441": 1, "9557226": 1, "93332666": 1, "9216192": 1, "0051875": 1, "1498923": 1, "919689": 1, "2018073": 1, "1340165": 1, "078974": 1, "3512629": 1, "9220221": 1, "3320776": 1, "6820607": 1, "0199469": 1, "7036613": 1, "106988": 1, "9190293": 1, "92025375": 1, "9720231": 1, "2163883": 1, "2135032": 1, "0164635": 1, "2299625": 1, "9603485": 1, "049892": 1, "956555": 1, "8135519": 1, "4731698": 1, "1520662": 1, "9206653": 1, "0245452": 1, "2051382": 1, "0095395": 1, "4394192": 1, "0033306": 1, "1800035": 1, "8209941": 1, "0029142": 1, "5246105": 1, "92345876": 1, "2150614": 1, "008857": 1, "0560567": 1, "9302045": 1, "7030569": 1, "1012498": 1, "7860416": 1, "0599829": 1, "4761962": 1, "202528": 1, "5128016": 1, "445789": 1, "4724422": 1, "1509331": 1, "92298305": 1, "0318545": 1, "9262946": 1, "8479152": 1, "9545922": 1, "0327728": 1, "9686907": 1, "938567": 1, "9457507": 1, "945941": 1, "2408772": 1, "91897124": 1, "98646605": 1, "9257557": 1, "120081": 1, "92295873": 1, "98197186": 1, "0966231": 1, "4087183": 1, "2270315": 1, "6620584": 1, "92366135": 1, "1687988": 1, "9332343": 1, "9522883": 1, "9668265": 1, "95776415": 1, "1525652": 1, "0056993": 1, "3006487": 1, "919276": 1, "5477448": 1, "223593": 1, "2821094": 1, "2692181": 1, "4564832": 1, "1377126": 1, "0408291": 1, "2340343": 1, "5079067": 1, "1427692": 1, "96081585": 1, "6995425": 1, "9474476": 1, "9666572": 1, "1406": 1, "752685546875": 1, "broadcast": 1, "version": 1, "in_ax": [1, 5], "tupl": [1, 4], "vmap": 1, "also": [1, 2, 5, 7], "exactli": 1, "ax": [1, 3], "arg": [1, 4], "should": 1, "in_arg": 1, "each": [1, 3, 5], "numpi": [1, 5], "jnp": [1, 5], "add_normal_nois": 1, "noise1": 1, "noise2": 1, "arr": 1, "ones": 1, "100": [1, 5], "subkei": [1, 5], "split": [1, 5], "101": 1, "arrai": [1, 2, 5], "_": [1, 3, 5], "maptrac": 1, "4996879": 1, "2619872": 1, "9215518": 1, "479387": 1, "1974081": 1, "3033798": 1, "8249735": 1, "0947304": 1, "9251105": 1, "0034034": 1, "9407734": 1, "9724346": 1, "92531097": 1, "3261323": 1, "1271875": 1, "9945743": 1, "6666276": 1, "0104258": 1, "0538225": 1, "7994913": 1, "92095697": 1, "1788874": 1, "1146896": 1, "7868929": 1, "0780694": 1, "9550102": 1, "0018265": 1, "92260957": 1, "9988542": 1, "9713293": 1, "3433127": 1, "1105025": 1, "3307436": 1, "2368753": 1, "656857": 1, "9746938": 1, "91894704": 1, "95909494": 1, "98636127": 1, "5532479": 1, "2469623": 1, "52319": 1, "99405545": 1, "3928486": 1, "1056511": 1, "8195016": 1, "106034": 1, "97066283": 1, "178679": 1, "91912156": 1, "3063895": 1, "2180374": 1, "1088842": 1, "1915169": 1, "96038926": 1, "681644": 1, "9702654": 1, "9791499": 1, "4835484": 1, "92048305": 1, "3628099": 1, "0404212": 1, "9202048": 1, "97043836": 1, "4648683": 1, "2770137": 1, "1500401": 1, "9911444": 1, "9267208": 1, "9934941": 1, "3678193": 1, "5602175": 1, "93110645": 1, "94659233": 1, "1082581": 1, "5226872": 1, "1722138": 1, "1196561": 1, "3667326": 1, "1623846": 1, "3700376": 1, "393898": 1, "192997": 1, "2921034": 1, "2701498": 1, "8569587": 1, "2345382": 1, "371684": 1, "9508094": 1, "91901976": 1, "0662607": 1, "92400897": 1, "0946498": 1, "077042": 1, "1119748": 1, "2475514": 1, "4764498": 1, "3618982": 1, "525895": 1, "3508108": 1, "1352432": 1, "633423": 1, "3747101": 1, "1413349": 1, "6587014": 1, "017165": 1, "99243104": 1, "233687": 1, "7807326": 1, "0443164": 1, "0233457": 1, "4869394": 1, "3447261": 1, "0242009": 1, "0232397": 1, "3417485": 1, "1536787": 1, "9463921": 1, "91894865": 1, "105353": 1, "1992693": 1, "1385094": 1, "9326704": 1, "8278294": 1, "97854596": 1, "341637": 1, "94762075": 1, "8663912": 1, "6927972": 1, "3196614": 1, "1932757": 1, "4879124": 1, "9352712": 1, "1170481": 1, "2921902": 1, "9379259": 1, "572887": 1, "9492164": 1, "3502445": 1, "94412285": 1, "4575214": 1, "0087024": 1, "0129974": 1, "92087346": 1, "8429991": 1, "1183012": 1, "92588425": 1, "1563082": 1, "264665": 1, "077333": 1, "92953223": 1, "0440084": 1, "945553": 1, "96677864": 1, "2360464": 1, "2453454": 1, "9212095": 1, "9724591": 1, "0338527": 1, "0710133": 1, "7905726": 1, "2739587": 1, "2617623": 1, "4346546": 1, "9396507": 1, "4231015": 1, "4537714": 1, "9463208": 1, "9230282": 1, "2071595": 1, "8987889": 1, "0588609": 1, "1534215": 1, "9482748": 1, "9277003": 1, "1802154": 1, "92822033": 1, "5889597": 1, "44008": 1, "0569656": 1, "3922757": 1, "0487039": 1, "9714392": 1, "9678032": 1, "0111053": 1, "0114919": 1, "92666644": 1, "115062": 1, "9226179": 1, "92357737": 1, "0207607": 1, "0289385": 1, "4577129": 1, "99169505": 1, "785027": 1, "97493356": 1, "4457805": 1, "490747": 1, "4962281": 1, "3267463": 1, "634931": 1, "89541": 1, "2962618": 1, "6207218": 1, "8561096": 1, "3205447": 1, "8174045": 1, "3284173": 1, "6": [1, 3], "705843": 1, "04772": 1, "9641192": 1, "459374": 1, "2700372": 1, "3503332": 1, "1504273": 1, "3363228": 1, "8203063": 1, "9568179": 1, "9727712": 1, "9048443": 1, "1202264": 1, "3173966": 1, "04736": 1, "6147223": 1, "0566154": 1, "296647": 1, "9494472": 1, "7890007": 1, "6916513": 1, "2909906": 1, "5365887": 1, "598415": 1, "2660148": 1, "3539233": 1, "949047": 1, "9126196": 1, "491834": 1, "9083114": 1, "3366058": 1, "4973707": 1, "7044837": 1, "5318923": 1, "007053": 1, "3137221": 1, "9486504": 1, "9378028": 1, "0319185": 1, "126971": 1, "443344": 1, "9964545": 1, "2359216": 1, "2620459": 1, "0544372": 1, "1582956": 1, "1964357": 1, "9269893": 1, "891475": 1, "9516089": 1, "5174012": 1, "9914963": 1, "1533825": 1, "31438": 1, "181967": 1, "405093": 1, "404519": 1, "7001152": 1, "6038115": 1, "9374652": 1, "849749": 1, "2006536": 1, "2666082": 1, "6190784": 1, "084528": 1, "8948672": 1, "0359583": 1, "7029026": 1, "100434": 1, "7086158": 1, "8068125": 1, "2193503": 1, "7623134": 1, "442602": 1, "164436": 1, "2599065": 1, "2812552": 1, "8684506": 1, "1612046": 1, "4867458": 1, "8734274": 1, "8425971": 1, "0870214": 1, "9529475": 1, "5523627": 1, "068737": 1, "8970017": 1, "222485": 1, "9222302": 1, "8526452": 1, "022123": 1, "677557": 1, "281": 1, "55023193359375": 1, "descend": 2, "mostli": 2, "inherit": 2, "concept": [2, 7], "refer": [2, 3], "from": [2, 3, 7], "few": [2, 3], "differ": 2, "stem": 2, "underli": 2, "section": 2, "describ": [2, 3], "sever": 2, "try": 2, "highlight": 2, "workaround": 2, "discuss": 2, "reason": [2, 5], "discrep": 2, "encod": [2, 3], "form": [2, 3], "unbound": 2, "recurs": 2, "becaus": 2, "doe": 2, "featur": 2, "mechan": 2, "dynam": 2, "alloc": 2, "abil": [2, 3, 5, 6, 7], "data": [2, 3, 5], "tension": 2, "know": [2, 3], "everyth": 2, "howev": 2, "gener": [2, 5, 6, 7], "function": [2, 5, 6, 7], "combin": [2, 4, 7], "bound": 2, "unfold": [2, 7], "direct": [2, 5], "pre": 2, "enough": [2, 3], "size": 2, "handl": [2, 3], "within": 2, "If": [2, 5, 7], "exceed": 2, "python": [2, 3, 4, 6], "error": 2, "thrown": 2, "both": 2, "practic": 2, "mean": [2, 5], "some": [2, 5], "engin": 2, "space": [2, 7], "v": [2, 3, 5], "julia": [2, 3, 7], "automat": 2, "virtu": 2, "being": 2, "top": [2, 6], "u": [2, 3, 6], "compat": [2, 4], "user": 2, "thu": 2, "idiom": [2, 5], "infer": [2, 3, 4, 5, 7], "small": 2, "diff": 2, "compar": [2, 5], "algorithm": [2, 4, 7], "object": [3, 7], "concis": [3, 7], "design": [3, 5], "customiz": 3, "bayesian": [3, 5], "formal": 3, "mathemat": 3, "represent": [3, 4], "probabilist": [3, 5, 7], "permit": 3, "structur": [3, 7], "captur": 3, "notion": 3, "exist": 3, "uncertainti": 3, "marco": [3, 7], "cusumano": [3, 7], "towner": [3, 7], "thesi": [3, 7], "knowledg": [3, 5], "One": [3, 6], "li": 3, "akin": 3, "languag": [3, 7], "reli": 3, "upon": 3, "intermedi": 3, "oper": 3, "transform": [3, 4], "speak": 3, "abov": 3, "you": [3, 5, 7], "d": 3, "jump": 3, "right": 3, "read": 3, "about": [3, 5], "visit": 3, "pure": 3, "roughli": 3, "subset": 3, "decor": 3, "jaxgenerativefunct": 3, "let": 3, "studi": [3, 5], "jaxpr": 3, "make_jaxpr": 3, "pretty_print": 3, "use_color": 3, "fals": 3, "lambda": [3, 5], "u32": 3, "b": 3, "c": [3, 7], "addr": 3, "see": 3, "someth": [3, 5], "keyword": 3, "doesn": 3, "t": [3, 4, 7], "nativ": [3, 6], "semant": [3, 4], "fry": 3, "random_wrap": 3, "impl": 3, "random_split": 3, "count": 3, "random_unwrap": 3, "e": 3, "slice": 3, "limit_indic": 3, "start_indic": 3, "stride": 3, "f": [3, 4, 7], "squeez": 3, "dimens": 3, "g": 3, "j": [3, 5, 6], "random_bit": 3, "bit_width": 3, "32": 3, "k": 3, "shift_right_log": 3, "9": 3, "l": 3, "1065353216": 3, "m": 3, "bitcast_convert_typ": 3, "new_dtyp": 3, "float32": 3, "n": 3, "sub": 3, "o": 3, "9999999403953552": 3, "p": [3, 4], "mul": 3, "q": 3, "add": 3, "r": [3, 4], "reshap": 3, "none": [3, 5], "new_siz": 3, "max": 3, "erf_inv": 3, "4142135381698608": 3, "w": 3, "y": 3, "z": 3, "ba": 3, "bb": 3, "bc": 3, "bd": 3, "div": 3, "bf": 3, "ab": 3, "bg": 3, "integer_pow": 3, "bh": 3, "log": 3, "283185307179586": 3, "bi": 3, "convert_element_typ": 3, "weak_typ": 3, "bj": 3, "bk": 3, "bl": 3, "bm": 3, "bn": 3, "bo": 3, "bp": 3, "reduce_sum": 3, "bq": 3, "That": 3, "quit": 3, "lot": 3, "new": [3, 7], "expand": 3, "sampl": [3, 4, 7], "updat": [3, 4], "prng": 3, "record": [3, 4], "probabl": 3, "densiti": 3, "result": 3, "piec": 3, "out": 3, "essenti": 3, "repeat": 3, "map": [4, 5, 7], "conceptu": 4, "core": 4, "expos": [4, 7], "when": 4, "kwarg": 4, "correspond": 4, "below": [4, 5], "given": [4, 7], "sim": 4, "cdot": 4, "appli": 4, "ret": 4, "along": 4, "evolv": 4, "well": [4, 5], "metadata": 4, "accru": 4, "dure": 4, "9247955083847046": 4, "It": [5, 7], "dead": 5, "simpl": 5, "often": 5, "first": 5, "framework": [5, 6, 7], "eight": 5, "school": 5, "problem": 5, "cover": 5, "extens": 5, "gelman": 5, "et": 5, "al": 5, "analysi": 5, "sec": 5, "2003": 5, "descript": 5, "wa": 5, "educ": 5, "test": 5, "servic": 5, "analyz": 5, "effect": 5, "special": 5, "coach": 5, "sat": 5, "scholast": 5, "aptitud": 5, "verbal": 5, "high": 5, "outcom": 5, "administr": 5, "standard": 5, "multipl": 5, "administ": 5, "help": 5, "colleg": 5, "make": 5, "admiss": 5, "decis": 5, "vari": 5, "between": 5, "200": 5, "800": 5, "500": 5, "deviat": 5, "examin": 5, "resist": 5, "short": 5, "term": 5, "effort": 5, "specif": 5, "toward": 5, "improv": 5, "instead": 5, "reflect": 5, "acquir": 5, "develop": 5, "year": 5, "nevertheless": 5, "consid": 5, "its": 5, "veri": 5, "success": 5, "increas": 5, "prior": 5, "believ": 5, "more": 5, "than": 5, "other": 5, "were": 5, "numpyro": 5, "context": 5, "idiomat": 5, "creat": 5, "sigma": 5, "plate": 5, "mu": 5, "tau": 5, "theta": 5, "ob": 5, "j_school": 5, "cauchi": 5, "ever": 5, "just": 5, "close": 5, "here": 5, "eight_school": 5, "8": 5, "build": 6, "take": 6, "advantag": 6, "deploy": 6, "capabl": [6, 7], "proper": 6, "opportun": 6, "convert": 6, "tf": 6, "readi": 6, "via": 6, "tensorflow": 6, "path": 6, "web": 6, "hardwar": 7, "acceler": 7, "programmat": 7, "what": 7, "repres": 7, "measur": 7, "differenti": 7, "mont": 7, "carlo": 7, "precis": 7, "mathematicali": 7, "formul": 7, "phd": 7, "re": 7, "tour": 7, "don": 7, "mind": 7, "perus": 7, "carefulli": 7, "craft": 7, "document": 7, "albeit": 7, "anoth": 7, "might": 7, "enjoi": 7, "do": 7, "look": 7, "dif": 7, "ture": 7, "univers": 7, "webassembli": 7}, "objects": {"genjax": [[1, 0, 0, "-", "combinators"], [4, 0, 0, "-", "interface"]], "genjax.combinators": [[1, 0, 0, "-", "map"], [1, 0, 0, "-", "unfold"]], "genjax.combinators.map": [[1, 1, 1, "", "MapCombinator"]], "genjax.combinators.unfold": [[1, 1, 1, "", "UnfoldCombinator"]], "genjax.interface": [[4, 2, 1, "", "simulate"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "function", "Python function"]}, "titleterms": {"expos": 0, "c": 0, "gener": [0, 1, 3, 4], "function": [0, 1, 3, 4], "A": [0, 5], "new": 0, "modul": 0, "combin": 1, "unfold": 1, "map": 1, "dif": 2, "gen": 2, "jl": 2, "ture": 2, "univers": 2, "To": 2, "jit": 2, "what": 3, "i": 3, "do": 3, "look": 3, "like": 3, "genjax": [3, 6], "interfac": 4, "tour": 5, "fundament": 5, "from": 6, "webassembli": 6, "index": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})