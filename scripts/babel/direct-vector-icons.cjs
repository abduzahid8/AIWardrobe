// Resolve named font imports to public per-family entry points so Metro does
// not bundle every font and glyph map. Helpers and namespace imports stay intact.
const families = new Set([
  'AntDesign', 'Entypo', 'EvilIcons', 'Feather', 'FontAwesome', 'FontAwesome5',
  'FontAwesome6', 'Fontisto', 'Foundation', 'Ionicons', 'MaterialCommunityIcons',
  'MaterialIcons', 'Octicons', 'SimpleLineIcons', 'Zocial',
]);
module.exports = function directVectorIcons({ types: t }) {
  return {
    visitor: {
      ImportDeclaration(path) {
        if (path.node.source.value !== '@expo/vector-icons' || path.node.importKind === 'type') return;
        const direct = [], remaining = [];
        for (const specifier of path.node.specifiers) {
          const name = t.isImportSpecifier(specifier) && specifier.imported.name;
          if (families.has(name) && specifier.importKind !== 'type') {
            direct.push(t.importDeclaration([t.importDefaultSpecifier(t.cloneNode(specifier.local))],
              t.stringLiteral(`@expo/vector-icons/${name}`)));
          } else remaining.push(specifier);
        }
        if (!direct.length) return;
        if (remaining.length) direct.unshift(t.importDeclaration(remaining, t.cloneNode(path.node.source)));
        path.replaceWithMultiple(direct);
      },
    },
  };
};
