const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder').pathfinder
const Movements = require('mineflayer-pathfinder').Movements
const { GoalFollow, GoalBlock, GoalNear } = require('mineflayer-pathfinder').goals
const express = require('express')

const MC_HOST = process.env.MC_HOST || 'localhost'
const MC_PORT = parseInt(process.env.MC_PORT || '49886')
const MC_USER = process.env.MC_USER || 'V_103'
const MC_VERSION = process.env.MC_VERSION || '1.20.1'
const API_PORT = parseInt(process.env.API_PORT || '3000')

const bot = mineflayer.createBot({
  host: MC_HOST,
  port: MC_PORT,
  username: MC_USER,
  version: MC_VERSION,
})

bot.loadPlugin(pathfinder)

// === STATE ===
let followTarget = null
let autoAttack = true
let autoFlee = true
let autoEat = true
let mode = 'idle'
let stayPos = null
let is_mining = false
let hunger_warning = false
let last_hunger_warning = 0
let loot_pos = null
let loot_timer = 0
let abort_task = false

// === FOOD LIST ===
const FOODS = [
  'bread','apple','cooked_beef','cooked_porkchop','cooked_chicken',
  'cooked_mutton','cooked_rabbit','baked_potato','carrot','golden_apple',
  'enchanted_golden_apple','beef','porkchop','chicken','mutton','rabbit',
  'potato','cookie','pumpkin_pie','cake','melon_slice','sweet_berries',
  'glow_berries','rotten_flesh','spider_eye'
]

function findFood() {
  return bot.inventory.items().find(i => FOODS.includes(i.name))
}

function setMode(newMode) {
  if (mode !== newMode) {
    console.log(`[MC] Mode: ${mode} → ${newMode}`)
    mode = newMode
  }
}

// === SPAWN ===
bot.on('spawn', () => {
  console.log(`[MC] V заспавнилась на ${MC_HOST}:${MC_PORT}`)
  const mcData = require('minecraft-data')(bot.version)
  const defaultMove = new Movements(bot, mcData)
  bot.pathfinder.setMovements(defaultMove)
})

bot.on('chat', (username, message) => {
  if (username === bot.username) return
  console.log(`[MC] Чат: <${username}> ${message}`)
})

bot.on('health', () => {
  if (bot.health < 10) {
    console.log(`[MC] V ранена! HP: ${bot.health}`)
  }
})

// === ENTITY DEAD → LOOT ===
bot.on('entityDead', (entity) => {
  if (entity.type === 'mob' && bot.entity.position.distanceTo(entity.position) < 5) {
    loot_pos = entity.position.clone()
    loot_timer = 100
  }
})

// === PHYSICS TICK ===
bot.on('physicsTick', () => {
  if (mode === 'follow' && followTarget && !is_mining) {
    const player = bot.players[followTarget]?.entity
    if (player) {
      if (!bot.pathfinder.goal) {
        const mcData = require('minecraft-data')(bot.version)
        const movements = new Movements(bot, mcData)
        bot.pathfinder.setMovements(movements)
        bot.pathfinder.setGoal(new GoalFollow(player, 2))
      }
    } else {
      followTarget = null
      setMode('idle')
    }
  }

  if (mode === 'stay' && stayPos && !is_mining) {
    if (bot.entity.position.distanceTo(stayPos) > 1.5) {
      if (!bot.pathfinder.goal || bot.pathfinder.goal.constructor.name !== 'GoalBlock') {
        bot.pathfinder.setGoal(new GoalBlock(stayPos.x, stayPos.y, stayPos.z))
      }
    }
  }

  if (autoFlee) {
    const creeper = bot.nearestEntity(e => e.name === 'creeper' && bot.entity.position.distanceTo(e.position) < 6)
    if (creeper) {
      bot.setControlState('back', true)
      bot.setControlState('jump', true)
      setTimeout(() => {
        bot.setControlState('back', false)
        bot.setControlState('jump', false)
      }, 600)
    }
  }

  if (autoAttack && bot.entity.onGround && mode !== 'mine') {
    const mob = bot.nearestEntity(e => {
      if (e.type !== 'mob') return false
      const dist = bot.entity.position.distanceTo(e.position)
      const hostile = ['zombie','skeleton','spider','cave_spider','enderman','witch','husk','drowned','phantom']
      return dist < 2.5 && hostile.includes(e.name)
    })
    if (mob) bot.attack(mob)
  }

  if (autoEat && bot.food < 5 && !is_mining) {
    const foodItem = findFood()
    if (foodItem) {
      bot.equip(foodItem, 'hand').then(() => {
        bot.consume().catch(() => {})
      }).catch(() => {})
    } else {
      const now = Date.now()
      if (now - last_hunger_warning > 90000) {
        last_hunger_warning = now
        hunger_warning = true
        setTimeout(() => { hunger_warning = false }, 45000)
      }
    }
  }

  if (loot_timer > 0) {
    loot_timer--
    const item = bot.nearestEntity(e => e.name === 'item' && bot.entity.position.distanceTo(e.position) < 5)
    if (item) {
      if (!bot.pathfinder.goal || bot.pathfinder.goal.constructor.name !== 'GoalBlock') {
        bot.pathfinder.setGoal(new GoalBlock(item.position.x, item.position.y, item.position.z))
      }
    }
    if (loot_timer <= 0) {
      loot_pos = null
      if (mode === 'idle') bot.pathfinder.setGoal(null)
    }
  }
})

// === ASYNC MINE ===
async function mineBlocks(targetBlock, count) {
  if (is_mining) {
    console.log('[MC] Уже копаю, игнорирую новую команду mine')
    return
  }
  is_mining = true
  abort_task = false
  setMode('mine')
  followTarget = null
  stayPos = null
  let mined = 0
  try {
    while (mined < count) {
      if (abort_task) {
        console.log('[MC] Добыча прервана')
        break
      }
      const mcData = require('minecraft-data')(bot.version)
      const block = bot.findBlock({
        matching: (block) => block.name.includes(targetBlock),
        maxDistance: 32
      })
      if (!block) {
        bot.chat(`*V не нашла ${targetBlock} рядом*`)
        break
      }
      if (abort_task) break
      const movements = new Movements(bot, mcData)
      bot.pathfinder.setMovements(movements)
      await bot.pathfinder.goto(new GoalBlock(block.position.x, block.position.y, block.position.z))
      if (abort_task) break
      await bot.dig(block)
      mined++
    }
  } catch (e) {
    console.log('[MC] Ошибка добычи:', e.message)
  }
  is_mining = false
  if (mode === 'mine') {
    setMode('idle')
  }
  console.log(`[MC] Добыча завершена. Добыто: ${mined}`)
}

// === HTTP API ===
const app = express()
app.use(express.json())

app.post('/command', async (req, res) => {
  const { action, target } = req.body
  let result = { ok: true }

  try {
    switch (action) {
      case 'follow': {
        const player = bot.players[target]?.entity
        if (player) {
          followTarget = target
          setMode('follow')
          stayPos = null
          abort_task = true
          bot.pathfinder.setGoal(null)
          const mcData = require('minecraft-data')(bot.version)
          const movements = new Movements(bot, mcData)
          bot.pathfinder.setMovements(movements)
          bot.pathfinder.setGoal(new GoalFollow(player, 2))
        } else {
          result = { ok: false, error: 'Игрок не найден' }
        }
        break
      }
      case 'stay': {
        stayPos = bot.entity.position.clone()
        setMode('stay')
        followTarget = null
        abort_task = true
        bot.pathfinder.setGoal(null)
        break
      }
      case 'stop': {
        followTarget = null
        setMode('idle')
        stayPos = null
        abort_task = true
        bot.pathfinder.setGoal(null)
        break
      }
      case 'attack': {
        const mob = bot.nearestEntity(e => e.name === target || (target === 'mob' && e.type === 'mob'))
        if (mob) bot.attack(mob)
        else result = { ok: false, error: 'Моб не найден' }
        break
      }
      case 'chat': {
        if (target) bot.chat(target)
        break
      }
      case 'jump': {
        bot.setControlState('jump', true)
        setTimeout(() => bot.setControlState('jump', false), 250)
        break
      }
      case 'look': {
        const player = bot.players[target]?.entity
        if (player) bot.lookAt(player.position)
        break
      }
      case 'collect': {
        const item = bot.nearestEntity(e => e.name === 'item' && bot.entity.position.distanceTo(e.position) < 12)
        if (item) {
          const mcData = require('minecraft-data')(bot.version)
          const movements = new Movements(bot, mcData)
          bot.pathfinder.setMovements(movements)
          bot.pathfinder.setGoal(new GoalBlock(item.position.x, item.position.y, item.position.z))
        }
        break
      }
      case 'equip': {
        const item = bot.inventory.items().find(i => i.name.includes(target))
        if (item) bot.equip(item, 'hand')
        break
      }
      case 'drop': {
        const item = bot.inventory.items().find(i => i.name.includes(target))
        if (item) bot.tossStack(item)
        break
      }
      case 'say_coords': {
        const p = bot.entity.position
        bot.chat(`Я на ${Math.floor(p.x)}, ${Math.floor(p.y)}, ${Math.floor(p.z)}`)
        break
      }
      case 'mine': {
        if (is_mining) {
          result = { ok: false, error: 'Уже копаю' }
          break
        }
        const parts = target.split(':')
        const blockName = parts[0]
        const count = parts.length > 1 ? parseInt(parts[1]) : 5
        if (!blockName) {
          result = { ok: false, error: 'Не указан блок для добычи' }
        } else if (isNaN(count) || count < 1) {
          result = { ok: false, error: 'Неверное количество' }
        } else {
          abort_task = false
          mineBlocks(blockName, count)
          result = { ok: true, message: `Начала добычу ${blockName} x${count}` }
        }
        break
      }
      case 'come': {
        const player = bot.players[target]?.entity
        if (player) {
          setMode('idle')
          followTarget = null
          stayPos = null
          abort_task = true
          bot.pathfinder.setGoal(null)
          const mcData = require('minecraft-data')(bot.version)
          const movements = new Movements(bot, mcData)
          bot.pathfinder.setMovements(movements)
          bot.pathfinder.setGoal(new GoalNear(player.position.x, player.position.y, player.position.z, 2))
        } else {
          result = { ok: false, error: 'Игрок не найден' }
        }
        break
      }
      case 'eat': {
        const foodItem = findFood()
        if (foodItem) {
          bot.equip(foodItem, 'hand').then(() => {
            bot.consume().catch(() => {})
          }).catch(() => {})
        } else {
          result = { ok: false, error: 'Еды нет' }
        }
        break
      }
      case 'give': {
        const parts = target.split(':')
        const itemName = parts[0]
        let count = parts.length > 1 ? (parts[1] === 'all' ? 9999 : parseInt(parts[1])) : 1
        if (!itemName) {
          result = { ok: false, error: 'Не указан предмет' }
        } else if (parts.length > 1 && parts[1] !== 'all' && (isNaN(count) || count < 1)) {
          result = { ok: false, error: 'Неверное количество' }
        } else {
          const item = bot.inventory.items().find(i => i.name.includes(itemName))
          if (item) {
            if (count >= item.count) {
              bot.tossStack(item)
            } else {
              bot.toss(item.type, null, count)
            }
          } else {
            result = { ok: false, error: 'Предмет не найден' }
          }
        }
        break
      }
      case 'emote': {
        const text = target || 'смотрит скептически'
        bot.chat(`*V ${text}*`)
        break
      }
      case 'info': {
        break
      }
      // === НОВЫЕ КОМАНДЫ ===
      case 'sleep': {
        const mcData = require('minecraft-data')(bot.version)
        const bed = bot.findBlock({
          matching: (block) => block.name.includes('bed') && block.name !== 'bedrock',
          maxDistance: 32
        })
        if (!bed) {
          result = { ok: false, error: 'Кровать не найдена' }
        } else {
          try {
            const movements = new Movements(bot, mcData)
            bot.pathfinder.setMovements(movements)
            await bot.pathfinder.goto(new GoalNear(bed.position.x, bed.position.y, bed.position.z, 1))
            await bot.sleep(bed)
            result = { ok: true, message: 'Легла спать' }
          } catch (e) {
            result = { ok: false, error: e.message }
          }
        }
        break
      }
      case 'wake': {
        try {
          await bot.wake()
          result = { ok: true, message: 'Проснулась' }
        } catch (e) {
          result = { ok: false, error: e.message }
        }
        break
      }
      case 'interact': {
        if (!target) {
          result = { ok: false, error: 'Не указан блок для взаимодействия' }
          break
        }
        const mcData = require('minecraft-data')(bot.version)
        const block = bot.findBlock({
          matching: (block) => block.name.includes(target),
          maxDistance: 32
        })
        if (!block) {
          result = { ok: false, error: `Блок ${target} не найден` }
        } else {
          try {
            const movements = new Movements(bot, mcData)
            bot.pathfinder.setMovements(movements)
            await bot.pathfinder.goto(new GoalNear(block.position.x, block.position.y, block.position.z, 2))
            await bot.activateBlock(block)
            result = { ok: true, message: `Взаимодействовала с ${block.name}` }
          } catch (e) {
            result = { ok: false, error: e.message }
          }
        }
        break
      }
      case 'craft': {
        const parts = target.split(':')
        const itemName = parts[0]
        const count = parts.length > 1 ? parseInt(parts[1]) : 1
        if (!itemName) {
          result = { ok: false, error: 'Не указан предмет для крафта' }
          break
        }
        if (isNaN(count) || count < 1) {
          result = { ok: false, error: 'Неверное количество' }
          break
        }
        try {
          const mcData = require('minecraft-data')(bot.version)
          const item = mcData.itemsByName[itemName]
          if (!item) {
            result = { ok: false, error: `Предмет ${itemName} не найден в данных игры` }
            break
          }
          let recipes = bot.recipesFor(item.id, null, 1, null)
          let craftingTable = null
          if (recipes.length === 0) {
            const tableBlock = bot.findBlock({
              matching: (b) => b.name.includes('crafting_table'),
              maxDistance: 32
            })
            if (tableBlock) {
              const tableRecipes = bot.recipesFor(item.id, null, 1, tableBlock)
              if (tableRecipes.length > 0) {
                recipes = tableRecipes
                craftingTable = tableBlock
              }
            }
          }
          if (recipes.length === 0) {
            result = { ok: false, error: `Рецепт для ${itemName} не найден или не хватает ресурсов` }
            break
          }
          const recipe = recipes[0]
          if (craftingTable) {
            const movements = new Movements(bot, mcData)
            bot.pathfinder.setMovements(movements)
            await bot.pathfinder.goto(new GoalNear(craftingTable.position.x, craftingTable.position.y, craftingTable.position.z, 2))
          }
          await bot.craft(recipe, count, craftingTable)
          result = { ok: true, message: `Скрафтила ${itemName} x${count}` }
        } catch (e) {
          result = { ok: false, error: e.message }
        }
        break
      }
      case 'blocks': {
        const blocks = []
        const bp = bot.entity.position
        for (let x = -4; x <= 4; x++) {
          for (let y = -2; y <= 2; y++) {
            for (let z = -4; z <= 4; z++) {
              const b = bot.blockAt(bp.offset(x, y, z))
              if (b && !['air','cave_air','void_air'].includes(b.name)) {
                blocks.push({
                  name: b.name,
                  pos: { x: b.position.x, y: b.position.y, z: b.position.z },
                  distance: Math.floor(bp.distanceTo(b.position))
                })
              }
            }
          }
        }
        blocks.sort((a, b) => a.distance - b.distance)
        result = { ok: true, blocks: blocks.slice(0, 20) }
        break
      }
      default:
        result = { ok: false, error: 'Unknown action' }
    }
  } catch (e) {
    result = { ok: false, error: e.message }
  }

  res.json(result)
})

app.get('/state', (req, res) => {
  const entities = Object.values(bot.entities)
    .filter(e => e.type === 'mob' || e.type === 'player')
    .map(e => ({
      name: e.name,
      type: e.type,
      pos: { x: Math.floor(e.position.x), y: Math.floor(e.position.y), z: Math.floor(e.position.z) },
      distance: Math.floor(bot.entity.position.distanceTo(e.position))
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 10)

  const mobs = entities.filter(e => e.type === 'mob' && e.distance < 20)

  const blocks = []
  const bp = bot.entity.position
  for (let x = -4; x <= 4; x++) {
    for (let y = -2; y <= 2; y++) {
      for (let z = -4; z <= 4; z++) {
        const b = bot.blockAt(bp.offset(x, y, z))
        if (b && !['air','cave_air','void_air'].includes(b.name)) {
          blocks.push({
            name: b.name,
            pos: { x: b.position.x, y: b.position.y, z: b.position.z },
            distance: Math.floor(bp.distanceTo(b.position))
          })
        }
      }
    }
  }
  blocks.sort((a, b) => a.distance - b.distance)

  res.json({
    connected: true,
    health: Math.floor(bot.health),
    food: Math.floor(bot.food),
    pos: {
      x: Math.floor(bot.entity.position.x),
      y: Math.floor(bot.entity.position.y),
      z: Math.floor(bot.entity.position.z)
    },
    inventory: bot.inventory.items().map(i => i.name),
    entities: entities,
    mobs: mobs,
    blocks: blocks.slice(0, 20),
    followTarget: followTarget,
    mode: mode,
    hunger_warning: hunger_warning,
    is_mining: is_mining,
    autoEat: autoEat
  })
})

app.listen(API_PORT, () => {
  console.log(`[MC] V bridge API on http://localhost:${API_PORT}`)
})