// Package rentalid owns the portable grammar of a Tensorhub-issued rental id.
// The id becomes both a URL segment and a local credential subject, so accepting
// path syntax or a Windows device name at either boundary would be unsafe.
package rentalid

import (
	"crypto/rand"
	"errors"
	"fmt"
	"math/big"
	"regexp"
	"strings"
)

var pattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
var machinePattern = regexp.MustCompile(`^[a-z][a-z0-9-]{0,31}$`)

// Valid reports whether id is one portable opaque name, not a path or device.
func Valid(id string) bool {
	if !pattern.MatchString(id) {
		return false
	}
	base := strings.ToUpper(strings.SplitN(id, ".", 2)[0])
	if base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" {
		return false
	}
	return !(len(base) == 4 &&
		(strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) &&
		base[3] >= '1' && base[3] <= '9')
}

// ValidMachineName reports whether a user-facing machine alias is short, shell-safe,
// and distinct from the reserved name for this computer.
func ValidMachineName(name string) bool {
	return name != "local" && machinePattern.MatchString(name)
}

// ErrNoFreeMachineName means every word in the vocabulary names a live rental.
var ErrNoFreeMachineName = errors.New("every machine word is in use by a live rental")

// NewMachineName draws one word this owner is not already using. A machine name
// only has to be unambiguous among the owner's own live rentals — Tensorhub's
// identity for the rental is its `pr-…` id — so a single memorable word is enough,
// and a word comes back into the draw once its rental is released.
func NewMachineName(taken map[string]bool) (string, error) {
	free := make([]string, 0, len(words))
	for _, word := range words {
		if !taken[word] {
			free = append(free, word)
		}
	}
	if len(free) == 0 {
		return "", ErrNoFreeMachineName
	}
	index, err := rand.Int(rand.Reader, big.NewInt(int64(len(free))))
	if err != nil {
		return "", fmt.Errorf("mint private rental name: %w", err)
	}
	return free[index.Int64()], nil
}

// Words is the whole machine vocabulary, sorted.
func Words() []string {
	return append([]string(nil), words...)
}

// words is the vocabulary a machine name is drawn from: animals, foods, small objects,
// weather and myth. Every entry satisfies ValidMachineName, none is a brand or a slur,
// and no two are one letter apart, so a name survives being typed from memory.
var words = []string{
	"aardvark", "abacus", "acorn", "adder", "agate", "agouti", "albatross", "alligator", "almond",
	"amber", "amethyst", "amulet", "anchovy", "anemone", "angelfish", "anteater", "antelope",
	"antler", "anvil", "apricot", "arepa", "armadillo", "astrolabe", "atoll", "aurora", "avalanche",
	"avocado", "avocet", "axle", "axolotl", "baboon", "badger", "bagel", "baguette", "baklava",
	"banana", "bard", "barley", "barnacle", "basalt", "basil", "basilisk", "bauble", "beacon",
	"bead", "beagle", "beetle", "behemoth", "bellows", "beluga", "beryl", "betta", "bilby", "biscuit",
	"bison", "bittern", "blaze", "blizzard", "boba", "bobbin", "bobcat", "bonito", "bonobo",
	"boulder", "breeze", "brioche", "broccoli", "brownie", "budgie", "buffalo", "bulgur", "bullfrog",
	"bumblebee", "bunting", "burrito", "butter", "butterfly", "button", "buzzard", "cabbage",
	"caiman", "calico", "camel", "candle", "cannoli", "canyon", "capuchin", "capybara", "caracal",
	"caramel", "cardinal", "caribou", "cashew", "cassava", "cassowary", "caterpillar", "centaur",
	"centipede", "chameleon", "cheddar", "cheetah", "chestnut", "chickpea", "chimera", "chinchilla",
	"chipmunk", "chisel", "chital", "churro", "cicada", "cichlid", "cider", "cirrus", "citrine",
	"clamp", "clove", "clownfish", "coati", "cobble", "cobra", "cocoa", "coconut", "collie",
	"colobus", "comet", "compass", "conker", "cookie", "coral", "corgi", "cosmos", "couscous",
	"coyote", "crag", "crank", "crepe", "cricket", "crocodile", "croissant", "crouton", "crumpet",
	"cuckoo", "cumin", "cumulus", "cupcake", "curlew", "custard", "cyclone", "cyclops", "dachshund",
	"damselfly", "dawn", "dingo", "dolphin", "domino", "donkey", "donut", "doodad", "dormouse",
	"doubloon", "dragon", "dragonfly", "drizzle", "dromedary", "druid", "dryad", "ducat", "dumpling",
	"dune", "earthworm", "echidna", "eclair", "eclipse", "egret", "eland", "elephant", "emerald",
	"empanada", "espresso", "falafel", "falcon", "farro", "farthing", "faun", "feather", "fennec",
	"fenrir", "ferret", "feta", "finch", "firefly", "flame", "flamingo", "flicker", "flint",
	"flipper", "florin", "flounder", "flurry", "foal", "focaccia", "forge", "fringe", "frost",
	"fudge", "gadget", "galago", "gale", "garlic", "garnet", "gazelle", "gecko", "gelato", "gerbil",
	"geyser", "gharial", "ginger", "giraffe", "gizmo", "glacier", "glade", "glimmer", "glowworm",
	"gnocchi", "gnome", "goat", "goblin", "goldfish", "golem", "goose", "gopher", "gorilla",
	"gouda", "gourami", "granite", "granola", "grebe", "gremlin", "greyhound", "griffin", "grizzly",
	"grosbeak", "grouper", "grouse", "grove", "guacamole", "guava", "gumdrop", "guppy", "gyoza",
	"haddock", "halibut", "hammer", "hamster", "harrier", "hazelnut", "heath", "hedgehog", "heron",
	"herring", "hinge", "hippo", "honey", "honeybee", "horizon", "hornet", "howler", "hummingbird",
	"hummus", "husky", "hydra", "ibex", "ibis", "iceberg", "iguana", "impala", "indri", "jackal",
	"jackdaw", "jade", "jasper", "jellybean", "jellyfish", "jerboa", "jester", "jewel", "kakapo",
	"kangaroo", "katydid", "kelpie", "kestrel", "kiln", "kimchi", "kingfisher", "kinkajou", "kitsune",
	"kiwi", "knapsack", "knight", "koala", "kodiak", "komodo", "kraken", "krill", "kudu", "kumquat",
	"ladybug", "lagoon", "lamb", "lamprey", "langur", "lantern", "latch", "latte", "ledger",
	"lemming", "lemonade", "lemur", "lentil", "leopard", "lettuce", "lever", "leviathan", "limpet",
	"linnet", "lionfish", "lobster", "locket", "lollipop", "loom", "lorikeet", "loris", "lychee",
	"macaque", "macaron", "macaw", "mackerel", "magpie", "mallard", "mallet", "mamba", "mammoth",
	"manatee", "mandrill", "mango", "manta", "manticore", "mantis", "marble", "marlin", "marmalade",
	"marmoset", "marmot", "marsh", "marshmallow", "marten", "mastiff", "matcha", "mayfly", "meadow",
	"meerkat", "merganser", "mermaid", "mesa", "meteor", "milkshake", "millipede", "minnow",
	"minotaur", "miso", "mochi", "monarch", "mongoose", "monsoon", "moorhen", "mouflon", "muntjac",
	"muskrat", "mussel", "naan", "naiad", "narwhal", "nautilus", "nebula", "nectar", "needle",
	"newt", "nickel", "nightjar", "nimbus", "ninja", "noodle", "nougat", "nova", "nugget", "numbat",
	"nuthatch", "nutmeg", "nymph", "oasis", "obsidian", "ocelot", "octopus", "ogre", "okapi",
	"olive", "omelet", "onyx", "opal", "orangutan", "oriole", "osprey", "ostrich", "otter", "oyster",
	"paella", "paladin", "pancake", "panda", "paneer", "pangolin", "panther", "papaya", "paprika",
	"parchment", "parrot", "parsley", "peanut", "pearl", "pebble", "pecan", "pegasus", "pelican",
	"pendant", "penguin", "peridot", "pesto", "petal", "petrel", "pheasant", "phoenix", "pickle",
	"pierogi", "piglet", "pinecone", "pipit", "piranha", "pirate", "pistachio", "pita", "pixie",
	"platypus", "plover", "plume", "polecat", "polenta", "pollock", "pomelo", "pompom", "pony",
	"popcorn", "porcupine", "porridge", "pouch", "prairie", "praline", "prawn", "pretzel", "prism",
	"pudding", "pufferfish", "puffin", "pulley", "pulsar", "pumice", "pumpkin", "quail", "quartz",
	"quasar", "quesadilla", "quill", "quince", "quinoa", "quokka", "rabbit", "raccoon", "radish",
	"rainbow", "ranger", "rasp", "raven", "reindeer", "rhino", "ribbon", "ripple", "risotto",
	"rivet", "robin", "rosemary", "ruby", "saffron", "salmon", "salsa", "sambar", "samosa", "samurai",
	"sandpiper", "sapphire", "sardine", "satchel", "satyr", "savanna", "scallop", "scone", "scorpion",
	"scroll", "seahorse", "selkie", "serval", "sextant", "shallot", "shilling", "shimmer", "shogun",
	"shrew", "shrike", "shrimp", "shuttle", "sifaka", "silkworm", "siren", "siskin", "skunk",
	"sleet", "sloth", "smoothie", "snail", "snapper", "sorbet", "sorcerer", "spaniel", "spanner",
	"spark", "sparkle", "sparrow", "sphinx", "spinach", "spindle", "spoke", "spool", "spoonbill",
	"sprocket", "sprout", "squall", "squid", "squire", "squirrel", "stallion", "staple", "starfish",
	"starling", "stingray", "stoat", "stork", "strudel", "sturgeon", "summit", "sunbeam", "sundae",
	"sunfish", "sunrise", "sunset", "sushi", "swallow", "swallowtail", "swordfish", "tabby",
	"taco", "tadpole", "taiga", "takin", "talisman", "talon", "tamale", "tamarin", "tanager",
	"tanuki", "tapioca", "tapir", "tarantula", "tarsier", "tassel", "telescope", "tempest", "tengu",
	"terrapin", "terrier", "thicket", "thimble", "thrush", "thunder", "thyme", "tiramisu", "titan",
	"toffee", "tofu", "tongs", "topaz", "tornado", "tortilla", "tortoise", "toucan", "treefrog",
	"trinket", "trout", "truffle", "tundra", "turnip", "turtle", "twilight", "twine", "twinkle",
	"typhoon", "unicorn", "urchin", "valkyrie", "vanilla", "viking", "viper", "waffle", "wagtail",
	"wahoo", "wallaby", "walnut", "walrus", "warbler", "warlock", "warthog", "wasabi", "waxwing",
	"weasel", "weevil", "whippet", "whirlpool", "whisker", "widget", "wildebeest", "wizard",
	"wolverine", "wombat", "wonton", "wren", "wrench", "wyvern", "yarn", "yeti", "yuzu", "zebra",
	"zenith", "zephyr", "zircon", "zucchini",
}
